import copy
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Union

from explainitall.clusters import Cluster, ClusterManager

UNNAMED_CLUSTER = 'unnamed'
DEFAULT_REPLACEMENT = '[...]'

# Слово: буквы и цифры, допускается дефис внутри («чуть-чуть», «кто-то»)
WORD_REGEX = re.compile(r'[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)*')

Messages = Union[str, List[dict]]


@dataclass(frozen=True)
class WordSpan:
    start: int
    end: int
    word: str


def find_words(text: str) -> List[WordSpan]:
    return [WordSpan(m.start(), m.end(), m.group()) for m in WORD_REGEX.finditer(text)]


def replace_spans(text: str, spans: Iterable[WordSpan], replacement: str = DEFAULT_REPLACEMENT) -> str:
    """Замена слов в тексте; при пустой замене слова удаляются вместе с лишними пробелами"""
    result, pos = [], 0
    for span in sorted(spans, key=lambda s: s.start):
        result.append(text[pos:span.start])
        result.append(replacement)
        pos = span.end
    result.append(text[pos:])
    new_text = ''.join(result)
    if replacement == '':
        new_text = re.sub(r'[ \t]{2,}', ' ', new_text)
        new_text = re.sub(r'[ \t]+([,.;:!?])', r'\1', new_text)
        new_text = re.sub(r'(^|\n)[ \t]+', r'\1', new_text)
        new_text = re.sub(r'[ \t]+(\n|$)', r'\1', new_text)
    return new_text


def occlusion_variants(text: str, span_groups: Sequence[Sequence[WordSpan]],
                       replacement: str = DEFAULT_REPLACEMENT, empty_text: str = '.') -> List[str]:
    """
    Варианты текста, в каждом из которых скрыта одна группа слов.
    Если после удаления не осталось текста, подставляется empty_text: API не принимает пустые строки.
    """
    variants = []
    for spans in span_groups:
        variant = replace_spans(text, spans, replacement)
        variants.append(variant if variant.strip() else empty_text)
    return variants


class TextOcclusion:
    """Группы слов текста для исключения: каждое слово отдельно или все слова кластера вместе"""

    def __init__(self, nlp_model=None, replacement: str = ''):
        """
        nlp_model: векторная модель слов gensim; нужна только для кластеров
        replacement: чем заменять скрытые слова; пустая строка — удалить
        """
        self.replacement = replacement
        self._cluster_manager = ClusterManager(embeddings=nlp_model) if nlp_model is not None else None

    @staticmethod
    def word_groups(text: str):
        """(слова, группы позиций): по одной группе на каждое слово"""
        spans = find_words(text)
        return [s.word for s in spans], [[s] for s in spans]

    def cluster_groups(self, text: str, clusters_description: List[dict]):
        """(имена кластеров, найденные слова, группы позиций) для кластеров, слова которых есть в тексте"""
        if self._cluster_manager is None:
            raise ValueError('Для кластеров нужен nlp_model')
        clusters = self._cluster_manager.create_clusters(clusters_description)
        masker = ClusterMasker(self._cluster_manager, clusters, replacement=self.replacement)
        spans = masker.find_cluster_spans(text)
        names = [name for name in masker.cluster_names if spans[name]]
        return names, [[s.word for s in spans[n]] for n in names], [spans[n] for n in names]

    def variants(self, text: str, groups: Sequence[Sequence[WordSpan]]) -> List[str]:
        return occlusion_variants(text, groups, self.replacement)


def normalize_messages(messages: Messages) -> List[dict]:
    if isinstance(messages, str):
        return [{'role': 'user', 'content': messages}]
    return copy.deepcopy(list(messages))


class ClusterMasker:
    """
    Поиск слов кластеров в промпте и построение промптов без них.
    Слово относится к первому кластеру, в который входит его нормальная форма.
    """

    def __init__(self, cluster_manager: ClusterManager, clusters: Sequence[Cluster],
                 roles: Iterable[str] = ('user',), replacement: str = DEFAULT_REPLACEMENT):
        """
        roles: роли сообщений, в которых ищутся и скрываются слова (по умолчанию только запрос пользователя)
        replacement: чем заменять слова кластера; пустая строка — удалить слово
        """
        self.cluster_manager = cluster_manager
        self.clusters = list(clusters)
        self.roles = set(roles)
        self.replacement = replacement

    @property
    def cluster_names(self) -> List[str]:
        return [c.name for c in self.clusters]

    def cluster_of(self, word: str) -> str:
        return self.cluster_manager.find_cluster_name(word, self.clusters)

    def find_cluster_spans(self, text: str) -> Dict[str, List[WordSpan]]:
        spans = {name: [] for name in self.cluster_names}
        for span in find_words(text):
            name = self.cluster_of(span.word)
            if name != UNNAMED_CLUSTER:
                spans[name].append(span)
        return spans

    def find_cluster_words(self, messages: Messages) -> Dict[str, List[str]]:
        """Слова каждого кластера, найденные в сообщениях с ролями self.roles"""
        found = {name: [] for name in self.cluster_names}
        for message in normalize_messages(messages):
            if message.get('role') in self.roles and isinstance(message.get('content'), str):
                for name, spans in self.find_cluster_spans(message['content']).items():
                    found[name].extend(s.word for s in spans)
        return found

    def mask_cluster(self, messages: Messages, cluster_name: str) -> List[dict]:
        """Копия сообщений, в которой скрыты все слова кластера cluster_name"""
        return self.mask_clusters(messages, [cluster_name])

    def mask_clusters(self, messages: Messages, cluster_names: Iterable[str]) -> List[dict]:
        """Копия сообщений, в которой скрыты все слова перечисленных кластеров"""
        cluster_names = list(cluster_names)
        unknown = [name for name in cluster_names if name not in self.cluster_names]
        if unknown:
            raise KeyError(f'Неизвестные кластеры: {unknown!r}')
        masked = normalize_messages(messages)
        for message in masked:
            if message.get('role') in self.roles and isinstance(message.get('content'), str):
                spans = self.find_cluster_spans(message['content'])
                selected = [span for name in cluster_names for span in spans[name]]
                message['content'] = replace_spans(message['content'], selected, self.replacement)
        return masked

    def cluster_presence(self, text: str) -> Dict[str, int]:
        """Есть ли в тексте слова каждого кластера: 1 или 0"""
        return {name: int(bool(spans)) for name, spans in self.find_cluster_spans(text).items()}
