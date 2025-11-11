import pytest

from tests.bpe.tokenizer_v1 import count_pair, merge, train_bpe
from bpe_tokenizer_param import BPETokenizerParams

def test_train_bpe_one_merge_end_to_end():
    # маленький приклад: "abab" -> найбільш часта пара (a,b)
    params = train_bpe("abab", num_merges=1)
    # params має тип BPETokenizerParams
    assert isinstance(params, BPETokenizerParams)
    # після одного merge має з'явитись один запис у merges
    assert len(params.merges) == 1
    # розмір вокабу збільшився (щонайменше на 1)
    assert len(params.vocab) >= 256 + 1

def test_train_bpe_zero_merges():
    params = train_bpe("abab", num_merges=0)
    # без merges — merges має бути порожнім
    assert len(params.merges) == 0
    # vocab існує
    assert isinstance(params.vocab, dict)

def test_train_bpe_num_merges_more_than_pairs():
    params = train_bpe("abab", num_merges=10)
    assert len(params.merges) == 2
    assert isinstance(params.vocab, dict)

def test_count_pairs_simple():
    indices = [116, 104, 18, 116, 104]
    counts = count_pair(indices)
    assert counts[(116, 104)] == 2
    assert counts[(104, 18)] == 1
    assert counts[(18, 116)] == 1

    # інша пара відсутня
    assert counts.get((18, 104), 0) == 0

def test_count_no_pairs():
    indices = [116, 104, 18, 116]
    counts = count_pair(indices)
    assert counts[(116, 104)] == 1
    assert counts[(104, 18)] == 1
    assert counts[(18, 116)] == 1

def test_count_empty_indicies():
    indices = []
    counts = count_pair(indices)
    assert len(counts) == 0 

def test_merge_single_occurrence_in_the_end():
    # Приклад: [a, b, c] merge (b,c) -> new
    a, b, c = 99, 96, 12
    indices = [a, b, c]
    new_index = 108
    out = merge(indices, (b, c), new_index)
    assert out == [a, new_index]  # (b,c) замінено на new_index

def test_merge_single_occurrence_in_the_middle():
    a, b, c, d = 99, 96, 12, 198
    indices = [a, b, c, d]
    new_index = 108
    out = merge(indices, (b, c), new_index)
    assert out == [a, new_index, d]

def test_merge_no_occurrence():
    a, b, c = 99, 96, 12
    indices = [a, b, c]
    new_index = 108
    out = merge(indices, (100, 101), new_index)
    assert out == indices # no changes

def test_vocab_update():
    # vocab[97] = b'a', vocab[98] = b'b'
    params = BPETokenizerParams(vocab={ord('a'): b'a', ord('b'): b'b'}, merges={})
    index1, index2 = ord('a'), ord('b')
    new_index = 256
    params.vocab[new_index] = params.vocab[index1] + params.vocab[index2]
    assert params.vocab[new_index] == b'ab'