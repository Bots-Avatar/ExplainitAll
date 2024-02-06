import re
from typing import Tuple

import numpy as np
import pandas as pd
from inseq import FeatureAttributionOutput
from inseq.utils import sum_normalize_attributions

from explainitall import stat_helpers


def fix_ig_tokens(feature_attr: FeatureAttributionOutput):
    from transformers import GPT2Tokenizer
    from copy import deepcopy
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

        # ta1 = normalize_attributions(ta)
        # ta1 = np.sum(np.array(ta1, dtype=float), axis=2) / 100

        ta2 = sum_normalize_attributions(ta)
        ta2 = np.array(ta2, dtype=float)

        target_arrays.append(np.array(ta2, dtype=float))
    return target_arrays


class AttrObj:
    def __init__(self,
                 phrase_input: str,
                 phrase_generated: str,
                 tokens_input: Tuple[str],
                 tokens_generated: Tuple[str],
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


#
# ParsedAttribution = namedtuple("ParsedAttribution",
#                                ("generated_labels_only",
#                                 "generated_labels_full",
#                                 "array",
#                                 "input_text",
#                                 "generated_text_only"))


def attr_to_df(attr: AttrObj):
    """Преобразует атрибуты в DataFrame"""
    df = pd.DataFrame(attr.array)
    df.columns = attr.tokens_generated
    df = df.sort_index()
    df.insert(0, 'Tokens', attr.tokens_input + attr.tokens_generated)
    return df


#
# def group_by(attr, gauss_norm=False):
#     """
#     Группировка по словам и применение агрегации максимума для каждой группы.
#
#     :param attr: объект ParsedAttribution с данными для группировки
#     :param gauss_norm: флаг для применения нормализации с использованием гауссовых параметров
#     :return: объект ParsedAttribution с результатами группировки
#     """
#     input_grouped = Detokenizer(attr.input_text + attr.generated_text_only,
#                                 attr.generated_labels_full).group_text()
#     generated_grouped = Detokenizer(attr.generated_text_only, attr.generated_labels_only).group_text()
#
#     input_grouped_gb = []
#     for i, gr in enumerate(input_grouped):
#         input_grouped_gb += ([i] * len(gr))
#
#     generated_grouped_gb = []
#     for i, gr in enumerate(generated_grouped):
#         generated_grouped_gb += ([i] * len(gr))
#
#     grouped_columns = {}
#     for col, group in zip(attr.array.T, generated_grouped_gb):
#         if group in grouped_columns:
#             grouped_columns[group].append(col)
#         else:
#             grouped_columns[group] = [col]
#     result = {group: np.max(np.column_stack(cols), axis=1) for group, cols in grouped_columns.items()}
#     result_arr = np.column_stack([result[group] for group in sorted(result)])
#
#     df = pd.DataFrame(result_arr, index=input_grouped_gb)
#
#     # Group by the index (which is based on list1) and apply the max aggregation function
#     grouped_max = df.groupby(df.index).max()
#
#     # Convert the result back to a numpy array
#     result = grouped_max.to_numpy()
#     generated_labels_only = ["".join(x) for x in generated_grouped]
#     generated_labels_full = ["".join(x) for x in input_grouped]
#     if gauss_norm:
#         result = stat_helpers.calc_gauss_stat_params(result)['new_arr']
#
#     return ParsedAttribution(generated_labels_only=generated_labels_only,
#                              generated_labels_full=generated_labels_full,
#                              array=result,
#                              input_text=attr.input_text,
#                              generated_text_only=attr.generated_text_only)


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

        Аргументы:
        - s (str): исходный текст

        Возвращает:
        - str: очищенный текст
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

        Возвращает:
        - list: список группированных слов
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
    tokens_input_grouped = Detokenizer(attr.phrase_input,
                                       attr.tokens_input).group_text()

    tokens_generated_grouped = Detokenizer(attr.phrase_generated,
                                           attr.tokens_generated).group_text()

    tokens_input_generated_mask = calculate_mask(tokens_input_grouped + tokens_generated_grouped)
    tokens_generated_mask = calculate_mask(tokens_generated_grouped)

    squashed_array = squash_arr(attr.array,
                                squash_col_mask=tokens_generated_mask,
                                squash_row_mask=tokens_input_generated_mask)

    tokens_input_grouped_flatten = tuple("".join(x) for x in tokens_input_grouped)
    tokens_generated_grouped_flatten = tuple("".join(x) for x in tokens_generated_grouped)

    if gmm_norm:
        squashed_array = stat_helpers.calc_gmm_stat_params(squashed_array)['new_arr']

    return AttrObj(phrase_input=attr.phrase_input,
                   phrase_generated=attr.phrase_generated,
                   tokens_input=tokens_input_grouped_flatten,
                   tokens_generated=tokens_generated_grouped_flatten,
                   array=squashed_array)
