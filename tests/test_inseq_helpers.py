import numpy as np
import pytest

from explainitall.gpt_like_interp import inseq_helpers

test_data = [
    (['Reference! site - a-bout Lorem-Ipsum,',
      ['  ref', '!ere_', 'nce((', '__site', 'a-bout__', '@Lorem', '*Ipsum'],
      [['Ref', 'ere', 'nce'], ['site'], ['a-bout'], ['Lorem-', 'Ipsum']]]),
    (['Reference! q321 - a-bout Lorem-Ipsum,',
      ['  ref', '!ere_', 'nce((', '__q321', 'a-bout__', '@Lorem', '*Ipsum'],
      [['Ref', 'ere', 'nce'], ['q321'], ['a-bout'], ['Lorem-', 'Ipsum']]]),
    (['giving information on its origins, as well ',
      ['🃯🃯giving🃯', 'ÐinformationÐ  ', 'on__', 'its', '__origin', '&&&s', 'as', 'well'],
      [['giving'], ['information'], ['on'], ['its'], ['origin', 's'], ['as'], ['well']]]),
    ([' as a random Lipsum generator.',
      ['as🃯🃯', 'a', 'Ðrandom', '🃯lipsum', '###generator!'],
      [['as'], ['a'], ['random'], ['Lipsum'], ['generator']]]),
    (['Папа у Васи чуть-чуть силён в Математике!',
      ['папа🃯🃯', 'у', 'Ðва', '🃯с', '###и!', '!чуть', '-', 'чуть', 'силён', '#в!', 'математике'],
      [['Папа'], ['у'], ['Ва', 'с', 'и'], ['чуть-', '', 'чуть'], ['силён'], ['в'], ['Математике']]])]


# Apply parametrization
@pytest.mark.parametrize("inp_text,inp_pairs,expected_rez", test_data)
def test_detokenizer(inp_text, inp_pairs, expected_rez):
    fact_rez = inseq_helpers.Detokenizer(inp_text, inp_pairs).group_text()
    assert fact_rez == expected_rez


@pytest.mark.parametrize("inp_text,inp_pairs,expected_rez", test_data)
def test_detokenizer_keeps_one_element_per_token(inp_text, inp_pairs, expected_rez):
    fact_rez = inseq_helpers.Detokenizer(inp_text, inp_pairs).group_text()
    assert sum(map(len, fact_rez)) == len(inp_pairs)
    assert all(isinstance(t, str) for group in fact_rez for t in group)


def test_detokenizer_hyphen_word_joins_to_string():
    grouped = inseq_helpers.Detokenizer('он чуть-чуть болен', [' он', ' чуть', '-', 'чуть', ' бол', 'ен']).group_text()
    assert ["".join(g) for g in grouped] == ['он', 'чуть-чуть', 'болен']


def test_detokenizer_hyphen_without_dash_token():
    grouped = inseq_helpers.Detokenizer('кто-то пришёл', ['кто', '-то', ' приш', 'ёл']).group_text()
    assert grouped == [['кто-', 'то'], ['приш', 'ёл']]


def test_detokenizer_keeps_tokens_missing_in_text():
    grouped = inseq_helpers.Detokenizer('кошка спит', [' кошка', ' xyz', ' спит', '<|endoftext|>']).group_text()
    assert grouped == [['кошка'], ['', 'спит'], ['']]


def test_group_by_hyphenated_words():
    tokens_input = (' он', ' чуть', '-', 'чуть')
    tokens_generated = (' бол', 'ен')
    array = np.arange(12, dtype=float).reshape(6, 2)
    attr = inseq_helpers.AttrObj(phrase_input='он чуть-чуть', phrase_generated=' болен',
                                 tokens_input=tokens_input, tokens_generated=tokens_generated, array=array)

    grouped = inseq_helpers.group_by(attr)

    assert grouped.tokens_input == ('он', 'чуть-чуть')
    assert grouped.tokens_generated == ('болен',)
    np.testing.assert_array_equal(grouped.array, np.array([[1.], [7.], [11.]]))


def test_group_by_raises_on_shape_mismatch():
    attr = inseq_helpers.AttrObj(phrase_input='кошка', phrase_generated=' спит',
                                 tokens_input=(' кошка',), tokens_generated=(' спит',),
                                 array=np.zeros((3, 1)))
    with pytest.raises(ValueError):
        inseq_helpers.group_by(attr)


def test_squash_arr():
    array = np.array([[4., 7., 3., 8., 6.],
                      [4., 9., 7., 4., 1.],
                      [2., 6., 0., 0., 7.],
                      [2., 5., 5., 4., 8.],
                      [1., 4., 6., 10., 4.]])

    squashed_arr = inseq_helpers.squash_arr(arr=array,
                                            squash_row_mask=[[0, 1], [1, 5]],
                                            squash_col_mask=[[0, 3], [3, 5]],
                                            aggr_f=np.max)

    expected = np.array([[7., 8.],
                         [9., 10.]])
    np.testing.assert_array_equal(squashed_arr, expected)


def test_calculate_mask():
    grouped_data = [[1, 2, 3], [11, 22], [333]]
    expected_output = [[0, 3], [3, 5], [5, 6]]
    assert inseq_helpers.calculate_mask(grouped_data) == expected_output
