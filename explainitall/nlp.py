from functools import lru_cache

import gensim
import pymorphy3

# Теги частей речи pymorphy -> теги UPOS, которыми размечены модели RusVectores.
# pymorphy приводит причастия и деепричастия к инфинитиву, а сравнительную степень к прилагательному,
# поэтому они отображаются в VERB и ADJ соответственно.
PYMORPHY_TO_UPOS = {
    'NOUN': 'NOUN',
    'ADJF': 'ADJ', 'ADJS': 'ADJ', 'COMP': 'ADJ',
    'VERB': 'VERB', 'INFN': 'VERB', 'PRTF': 'VERB', 'PRTS': 'VERB', 'GRND': 'VERB',
    'NUMR': 'NUM',
    'ADVB': 'ADV', 'PRED': 'ADV',
    'NPRO': 'PRON',
    'PREP': 'ADP',
    'CONJ': 'CCONJ',
    'PRCL': 'PART',
    'INTJ': 'INTJ',
}


class WordProcessor:

    def __init__(self, gensim_nlp_embeddings: gensim.models.keyedvectors.KeyedVectors):
        self._morph = pymorphy3.MorphAnalyzer()
        self.gensim_nlp_embeddings = gensim_nlp_embeddings

    @lru_cache(maxsize=None)
    def get_clean_word(self, word: str):
        return word.lower().strip()

    # @lru_cache(maxsize=None)
    def get_embeddable_words_batch(self, words: list):
        cleaned = [self.get_clean_word(w) for w in words]
        embeddable = [self.get_embeddable_word_or_none(w) for w in cleaned if w]
        return [w for w in embeddable if w]

    @lru_cache(maxsize=None)
    def get_morph_or_none(self, word: str):
        morphed = self._morph.parse(word)
        known_words = [x for x in morphed if x.is_known]
        if not known_words:
            return None
        return known_words[0]

    @lru_cache(maxsize=None)
    def get_normal_form_or_none(self, word: str):
        clean_word = self.get_clean_word(word)
        if not clean_word:
            return None
        word_morph = self.get_morph_or_none(clean_word)
        if not word_morph:
            return None
        return word_morph.normal_form

    @lru_cache(maxsize=None)
    def get_grammeme_or_none(self, word: str):
        """Часть речи слова в нотации UPOS (NOUN, VERB, ADJ, ...)"""
        clean_word = self.get_clean_word(word)
        if not clean_word:
            return None
        word_morph = self.get_morph_or_none(clean_word)
        if not word_morph:
            return None
        return PYMORPHY_TO_UPOS.get(word_morph.tag.POS)

    def get_embeddable_word_or_none(self, word: str):
        normal_form = self.get_normal_form_or_none(word)
        grammeme = self.get_grammeme_or_none(word)
        if not normal_form or not grammeme:
            return None
        candidates = [f'{normal_form}_{grammeme}']
        if grammeme == 'NOUN':
            # Имена собственные в моделях RusVectores размечены как PROPN
            candidates.append(f'{normal_form}_PROPN')
        for word_tagged in candidates:
            if word_tagged in self.gensim_nlp_embeddings:
                return word_tagged
        return None
