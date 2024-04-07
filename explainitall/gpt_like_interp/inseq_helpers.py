import re
from copy import deepcopy
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from inseq import FeatureAttributionOutput
from inseq.utils.typing import GranularSequenceAttributionTensor as Gast
from inseq.utils.typing import TokenSequenceAttributionTensor as Tsat
from torch.linalg import vector_norm

from explainitall import stat_helpers


def sum_normalize_attributions(
        attributions: Union[Gast, Tuple[Gast, Gast]],
        cat_dim: int = 0,
        norm_dim: Optional[int] = 0,
) -> Tsat:
    """
    Суммаризация и нормализация тензоров по dim_sum
    РезультатЖ матрица векторов строк
    """
    concat = False
    if isinstance(attributions, tuple):
        concat = True
        orig_sizes = [a.shape[cat_dim] for a in attributions]
        attributions = torch.cat(attributions, dim=cat_dim)
    else:
        orig_sizes = [attributions.shape[cat_dim]]
    attributions = vector_norm(attributions, ord=2, dim=-1)
    if norm_dim is not None:
        attributions = attributions / attributions.nansum(dim=norm_dim, keepdim=True)
    if len(attributions.shape) == 1:
        attributions = attributions.unsqueeze(0)
    if concat:
        attributions = attributions.split(orig_sizes, dim=cat_dim)
        return attributions[0], attributions[1]
    return attributions


def fix_ig_tokens(feature_attr: FeatureAttributionOutput):
    from transformers import GPT2Tokenizer

    feature_attr_conv = deepcopy(feature_attr)
    dt = GPT2Tokenizer.from_pretrained(feature_attr_conv.info['model_name'])

    for attr in feature_attr_conv.sequence_attributions:
        for token_holder in (attr.source, attr.target):
            for s in token_holder:
                try:
                    s.token = dt.convert_tokens_to_string(s.token)
                except KeyError:
                    pass
    return feature_attr_conv


def get_ig_tokens(feature_attr: FeatureAttributionOutput):
    rez = []
    for attr in feature_attr.sequence_attributions:
        rez.append((tuple(s.token for s in attr.source),
                    tuple(t.token for t in attr.target)))
    return rez


def get_ig_phrases(feature_attr: FeatureAttributionOutput):
    return tuple(zip(feature_attr.info['input_texts'],
                     feature_attr.info['generated_texts']))


def get_g_arrays(feature_attr: FeatureAttributionOutput):
    target_arrays = []
    for attr in feature_attr.sequence_attributions:
        ta = attr.target_attributions
        ta2 = sum_normalize_attributions(ta)
        ta2 = np.array(ta2, dtype=float)

        target_arrays.append(np.array(ta2, dtype=float))
    return target_arrays


class AttrObj:
    def __init__(self,
                 phrase_input: str,
                 phrase_generated: str,
                 tokens_input: Tuple[str, ...],
                 tokens_generated: Tuple[str, ...],
                 array: np.ndarray):
        self.phrase_input = phrase_input
        self.phrase_generated = phrase_generated
        self.tokens_input = tokens_input
        self.tokens_generated = tokens_generated
        self.array = array

    def __repr__(self):
        return (f"AttrObj({self.phrase_input=}, "
                f"{self.phrase_generated=}, "
                f"{self.tokens_input=}, "
                f"{self.tokens_generated=}, "
                f"{self.array.shape=})")


def get_first_attribute(feature_attr: FeatureAttributionOutput):
    fixed_attr = fix_ig_tokens(feature_attr)
    phrase_input, phrase_generated_full = get_ig_phrases(fixed_attr)[0]
    phrase_generated = phrase_generated_full[len(phrase_input):]
    tokens_input, tokens_generated_full = get_ig_tokens(fixed_attr)[0]
    tokens_generated = tokens_generated_full[len(tokens_input):]
    array = get_g_arrays(fixed_attr)[0]

    return AttrObj(phrase_input=phrase_input,
                   phrase_generated=phrase_generated,
                   tokens_input=tokens_input,
                   tokens_generated=tokens_generated,
                   array=array)


def attr_to_df(attr: AttrObj):
    """Преобразует атрибуты в DataFrame"""
    df = pd.DataFrame(attr.array)
    df.columns = attr.tokens_generated
    df = df.sort_index()
    df.insert(0, 'Tokens', attr.tokens_input + attr.tokens_generated)
    return df


def squash_arr(arr, squash_row_mask, squash_col_mask, aggr_f=np.max):
    # Apply the mask to the rows
    row_result = np.array([aggr_f(arr[start:end], axis=0)
                           for start, end in squash_row_mask])
    # Apply the mask to the columns
    col_result = np.array([aggr_f(row_result[:, start:end], axis=1)
                           for start, end in squash_col_mask]).T
    return col_result


class Detokenizer:
    """
    Класс для детокенизации (приведения токенов к словам).
    """
    #  список символов-тире
    dash_chars = list(map(chr, (45, 8211, 8212, 8722, 9472, 9473, 9476)))
    # регулярное выражение для поиска тире между буквами
    dash_regex = re.compile(r'(?<!\w)-|-(?!\w)')
    # регулярное выражение для удаления небуквенных символов, кроме тире и пробелов
    clean_regex = re.compile(r'[^a-zA-Zа-яА-ЯёЁ0-9:\?\s\-]+|\s+')

    def __init__(self, text, tokenized_text):
        """
            Атрибуты:
            - text (str): исходный текст
            - pairs (list): список токенов

        """
        self.text = text
        self.pairs = tokenized_text

    def clean_text(self, s: str):
        """
        Очищает текст от ненужных символов.

        Аргументы: исходный текст
        Возвращает: очищенный текст
        """
        for dash in self.dash_chars:
            s = s.replace(dash, "-")
        s = s.replace(":", " : ")
        s = s.replace("?", " ? ")
        s = self.dash_regex.sub(' ', s).strip()
        return self.clean_regex.sub(' ', s).strip()

    def group_text(self):
        """
        Группирует токены в слова.
        Возвращает: список группированных слов
        """
        text = self.clean_text(self.text)
        pairs = [(self.clean_text(k)) for k in self.pairs]
        temp = []
        result = []
        for txt in pairs:
            start_index = text.lower().find(txt.lower())
            if start_index != -1:
                end_index = start_index + len(txt)
                temp.append(text[start_index:end_index])
                text = text[end_index:]
                if not text or text[0] == " ":
                    result.append(temp)
                    temp = []
                elif text[0] == "-" and txt != "-":
                    temp.append(('-', 0))
                    text = text[1:]
        return result


def calculate_mask(grouped_elements):
    lengths = list(map(len, grouped_elements))
    ranges = []
    start_index = 0
    for length in lengths:
        end_index = start_index + length
        ranges.append([start_index, end_index])
        start_index = end_index
    return ranges


def group_by(attr: AttrObj, gmm_norm=False) -> AttrObj:
    tokens_input_grouped = Detokenizer(attr.phrase_input, attr.tokens_input).group_text()
    tokens_generated_grouped = Detokenizer(attr.phrase_generated, attr.tokens_generated).group_text()
    tokens_input_generated_mask = calculate_mask(tokens_input_grouped + tokens_generated_grouped)
    tokens_generated_mask = calculate_mask(tokens_generated_grouped)

    squashed_array = squash_arr(attr.array,
                                squash_col_mask=tokens_generated_mask,
                                squash_row_mask=tokens_input_generated_mask)

    tokens_input_grouped_flatten = tuple(["".join(x) for x in tokens_input_grouped])
    tokens_generated_grouped_flatten = tuple("".join(x) for x in tokens_generated_grouped)

    if gmm_norm:
        squashed_array = stat_helpers.calc_gmm_stat_params(squashed_array)['new_arr']

    return AttrObj(phrase_input=attr.phrase_input,
                   phrase_generated=attr.phrase_generated,
                   tokens_input=tokens_input_grouped_flatten,
                   tokens_generated=tokens_generated_grouped_flatten,
                   array=squashed_array)
