import numpy as np
import pytest

from explainitall.gpt_like_interp import inseq_helpers

test_data = [
    (['Reference! site - a-bout Lorem-Ipsum,',
      ['  ref', '!ere_', 'nce((', '__site', 'a-bout__', '@Lorem', '*Ipsum'],
      [['Ref', 'ere', 'nce'], ['site'], ['a-bout'], ['Lorem', ('-', 0), 'Ipsum']]]),
    (['Reference! q321 - a-bout Lorem-Ipsum,',
      ['  ref', '!ere_', 'nce((', '__q321', 'a-bout__', '@Lorem', '*Ipsum'],
      [['Ref', 'ere', 'nce'], ['q321'], ['a-bout'], ['Lorem', ('-', 0), 'Ipsum']]]),
    (['giving information on its origins, as well ',
      ['🃯🃯giving🃯', 'ÐinformationÐ  ', 'on__', 'its', '__origin', '&&&s', 'as', 'well'],
      [['giving'], ['information'], ['on'], ['its'], ['origin', 's'], ['as'], ['well']]]),
    ([' as a random Lipsum generator.',
      ['as🃯🃯', 'a', 'Ðrandom', '🃯lipsum', '###generator!'],
      [['as'], ['a'], ['random'], ['Lipsum'], ['generator']]]),
    (['Папа у Васи чуть-чуть силён в Математике!',
      ['папа🃯🃯', 'у', 'Ðва', '🃯с', '###и!', '!чуть', '-', 'чуть', 'силён', '#в!', 'математике'],
      [['Папа'], ['у'], ['Ва', 'с', 'и'], ['чуть', ('-', 0), '', 'чуть'], ['силён'], ['в'], ['Математике']]])]


# Apply parametrization
@pytest.mark.parametrize("inp_text,inp_pairs,expected_rez", test_data)
def test_detokenizer(inp_text, inp_pairs, expected_rez):
    fact_rez = inseq_helpers.Detokenizer(inp_text, inp_pairs).group_text()
    assert fact_rez == expected_rez


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
