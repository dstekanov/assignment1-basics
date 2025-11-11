import time         
import os
from typing import List, Dict  
from bpe_tokenizer_param_v2 import BPETokenizerParamsV2
import regex as re
from collections import Counter
from typing import Dict, List, Tuple
from typing import BinaryIO
import pathlib

import cProfile
import pstats

from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = re.compile(PAT)
SPECIAL_TOKENS_PATTERN = None

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParamsV2:
    with open(input_path, "rb") as f:
        num_processes = 10
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # TODO: The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # pre_counts = Counter()
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
        #     f.seek(start)
        #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #     counter = pretokenize(chunk, special_tokens)
        #     pre_counts.update(counter)

        args_list = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(num_processes) as pool:
            results = pool.map(pretokenize_chunk, args_list)

        pre_counts = Counter()
        for result in results:
            pre_counts.update(result)
    
    token_indicies, freqs = tokens_to_indicies_lists(pre_counts)

    vocab = build_initial_vocab()
    
    next_id = update_vocab_with_special_tokens(vocab, special_tokens)

    merges: Dict[Tuple[int, int], int] = {}

    # 300 - 257 = 43 merges
    num_merges = vocab_size - len(vocab)
    print("Number of merging: ", num_merges)

    for i in range(num_merges):
        pair_counts = count_pair_frequencies(token_indicies, freqs)

        best_pair, best_count = get_best_pair(vocab, pair_counts)       

        new_id = next_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_id
        next_id += 1

        # print(f"Merge #{i+1}: {best_pair} w/ count {best_count} -> {new_id}, bytes={vocab[new_id]!r}")

        token_indicies = [
            apply_merge_on_token(s, best_pair, new_id) for s in token_indicies
        ]

    merges_list = [(vocab[pair[0]], vocab[pair[1]]) for pair, _ in merges.items()]

    return BPETokenizerParamsV2(vocab, merges_list)

def get_best_pair(vocab, pair_counts):
    """
    Select the pair with highest frequency. 
    Ties are broken by lexicographic order of merged bytes.
    
    Returns:
        (best_pair, count): The selected pair and its frequency
    """

    best_pair = None
    best_count = -1
    best_bytes_tuple = None

    for pair, count in pair_counts.items():
        # Merged bytes:  (b' K', b'ell')
        merged_bytes = (vocab[pair[0]], vocab[pair[1]])

        if (best_pair is None or count > best_count or (count == best_count and merged_bytes > best_bytes_tuple)):
            best_count = count
            best_bytes_tuple = merged_bytes
            best_pair = pair

    return best_pair, best_count

# TODO: The function modifies vocab directly (side effect)
def update_vocab_with_special_tokens(vocab, special_tokens: list[str]):
    next_id = max(vocab.keys()) + 1

    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1
    
    return next_id

# ---------- Core functions ----------
def pretokenize(text: str, special_tokens: list[str]) -> Counter:
    """Return Counter of pre-tokens using GPT-2 regex pattern."""

    # Optimization 1
        # chunks = re.split("|".join(re.escape(token) for token in special_tokens), text)
    global SPECIAL_TOKENS_PATTERN
    if SPECIAL_TOKENS_PATTERN is None:
        SPECIAL_TOKENS_PATTERN = re.compile("|".join(re.escape(t) for t in special_tokens))
    chunks = re.split(SPECIAL_TOKENS_PATTERN, text)

    # Optimization 2
        # matches = []
        # for chunk in chunks:
        #     matches.extend(re.finditer(PAT, chunk))
        
        # return Counter(m.group() for m in matches)
    counter = Counter()
    for chunk in chunks:
        # counter.update(m.group() for m in re.finditer(PAT, chunk))
        counter.update(m.group() for m in PAT_COMPILED.finditer(chunk))
    
    return counter

def pretokenize_chunk(args):
     input_path, start, end, special_tokens = args
     
     with open(input_path, "rb") as f:
          f.seek(start)
          chunk = f.read(end - start).decode("utf-8", errors="ignore")
          counter = pretokenize(chunk, special_tokens)
     return counter

def build_initial_vocab() -> Dict[int, bytes]:
    """Create the initial byte-level vocabulary: id -> single-byte bytes."""
    return {i: bytes([i]) for i in range(256)}

def tokens_to_indicies_lists(pre_counts: Counter):
    token_indicies = []
    freqs = []
    for token, freq in pre_counts.items():
        b = token.encode("utf-8")
        token_indicies.append(list(b))
        freqs.append(freq)
    return token_indicies, freqs

def count_pair_frequencies(token_indicies: List[List[int]], freqs: List[int]) -> Counter:
    """Count how often each adjacent byte-pair occurs, weighted by token frequency."""
    counts = Counter()
    for indicies, f in zip(token_indicies, freqs):
        for i in range(len(indicies) - 1):
            counts[(indicies[i], indicies[i + 1])] += f
    return counts

def apply_merge_on_token(indicies: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace non-overlapping occurrences of pair (a,b) with new_id."""
    a, b = pair
    new_indicies = []
    i = 0
    while i < len(indicies):
        if i < len(indicies) - 1 and indicies[i] == a and indicies[i + 1] == b:
            new_indicies.append(new_id)
            i += 2
        else:
            new_indicies.append(indicies[i])
            i += 1
    
    # print("New indicies: ", new_indicies)

    return new_indicies

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


FIXTURES_PATH = "/Users/dstekanov/Documents/own_projects/assignment1-basics/data"

if __name__ == "__main__":

    start_time = time.time()

    special_tokens = ["<|endoftext|>"]
    input_path = FIXTURES_PATH + "/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 500
    
    print(f"Training BPE on: '{input_path}' with {vocab_size} vocab_size...")

    profiler = cProfile.Profile()
    profiler.enable()

    result = train_bpe(input_path, vocab_size, special_tokens)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    print("Vocab: ", result.vocab)
    print("Merges: ", result.merges)

    print("Execution time:", round(time.time() - start_time, 3), "seconds")

