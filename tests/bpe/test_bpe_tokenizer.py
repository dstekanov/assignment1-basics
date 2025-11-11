import pytest

from bpe_tokenizer_param import BPETokenizerParams
from bpe_tokenizer import BPETokenizer

def test_encode_decode_roundtrip():
    # Setup small vocab so decode knows bytes for each index including new merged index
    a, b = ord('a'), ord('b')
    vocab = {i: bytes([i]) for i in range(256)}
    # create a merged token 256 for 'ab'
    merged_index = 256
    vocab[merged_index] = b'ab'

    # merges: (a,b)->256, so encode should merge occurrences of a,b into 256 if present
    merges = {(a, b): merged_index}
    params = BPETokenizerParams(vocab=vocab, merges=merges)
    tokenizer = BPETokenizer(params)

    s = "ab"
    encoded = tokenizer.encode(s)
    # After encode, should be [256] (merged token)
    assert encoded == [merged_index]

    # decode must convert indices back to string
    decoded = tokenizer.decode(encoded)
    assert decoded == "ab"

def test_decode_requires_all_keys_present():
    # If vocab is missing an index referenced by indices, decode will fail (TypeError or KeyError)
    vocab = {i: bytes([i]) for i in range(256)}
    # intentionally omit a mapping
    del vocab[97]  # remove 'a' byte mapping

    params = BPETokenizerParams(vocab=vocab, merges={})
    tokenizer = BPETokenizer(params)

    with pytest.raises(Exception):
        tokenizer.decode([97])  # attempt to decode but vocab[97] missing => should raise
