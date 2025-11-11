import time         
import os
from typing import List, Dict  
try:
    from .bpe_tokenizer_param_v2 import BPETokenizerParamsV2
except ImportError:
    from bpe_tokenizer_param_v2 import BPETokenizerParamsV2
import regex as re
from collections import Counter
from typing import Dict, List, Tuple
from typing import BinaryIO
import pathlib

import cProfile
import pstats
import tracemalloc
import pickle

from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = re.compile(PAT)
SPECIAL_TOKENS_PATTERN = None

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParamsV2:
    with open(input_path, "rb") as f:
        num_processes = 10
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

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

    # Build initial pair counts and position index (cache)
    pair_counts, pair_positions = build_pair_index(token_indicies, freqs)
    
    for i in range(num_merges):
        best_pair, best_count = get_best_pair(vocab, pair_counts)
        
        new_id = next_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_id
        next_id += 1

        # Update pair counts incrementally using position index
        update_pair_counts_with_index(token_indicies, freqs, pair_counts, pair_positions, best_pair, new_id)

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

def apply_merge_and_get_affected(
    token_indicies: List[List[int]], 
    pair: Tuple[int, int], 
    new_id: int
) -> List[int]:
    """
    Apply merge to all tokens and return indices of affected tokens.
    
    Returns:
        List of token indices that were modified
    """
    affected = []
    
    for token_idx in range(len(token_indicies)):
        old_indicies = token_indicies[token_idx]
        new_indicies = apply_merge_on_token(old_indicies, pair, new_id)
        
        # Check if token changed
        if len(old_indicies) != len(new_indicies):
            token_indicies[token_idx] = new_indicies
            affected.append(token_idx)
    
    return affected
    
def get_pairs_from_token(indicies: List[int]) -> List[Tuple[int, int]]:
    """Extract all adjacent pairs from a token."""
    return [(indicies[i], indicies[i + 1]) for i in range(len(indicies) - 1)]

def build_pair_index(token_indicies: List[List[int]], freqs: List[int]) -> Tuple[Counter, Dict]:
    """
    Build initial pair counts and position index.
    
    Returns:
        pair_counts: Counter of pair frequencies
        pair_positions: {pair: set of token indices that contain this pair}
    """
    pair_counts = Counter()
    pair_positions = {}
    
    for token_idx, (indicies, freq) in enumerate(zip(token_indicies, freqs)):
        for i in range(len(indicies) - 1):
            pair = (indicies[i], indicies[i + 1])
            pair_counts[pair] += freq
            
            if pair not in pair_positions:
                pair_positions[pair] = set()
            pair_positions[pair].add(token_idx)
    
    return pair_counts, pair_positions

def update_pair_counts_with_index(
    token_indicies: List[List[int]],
    freqs: List[int],
    pair_counts: Counter,
    pair_positions: Dict[Tuple[int, int], set],
    merged_pair: Tuple[int, int],
    new_id: int
):
    """
    Incrementally update pair counts after a merge using position index.
    O(1) lookup of affected tokens instead of O(n) iteration.
    """
    # Get affected tokens directly from index (O(1)!)
    if merged_pair not in pair_positions:
        return
    
    affected_tokens = list(pair_positions[merged_pair])  # Copy to avoid modification during iteration
    
    # Remove merged pair from index
    del pair_positions[merged_pair]
    
    for token_idx in affected_tokens:
        indicies = token_indicies[token_idx]
        freq = freqs[token_idx]
        
        # Get old pairs before merge
        old_pairs = get_pairs_from_token(indicies)
        
        # Apply merge
        new_indicies = apply_merge_on_token(indicies, merged_pair, new_id)
        token_indicies[token_idx] = new_indicies
        
        # Get new pairs after merge
        new_pairs = get_pairs_from_token(new_indicies)
        
        # Update counts: remove old, add new
        for pair in old_pairs:
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            
            # Update position index: remove this token from old pair
            if pair in pair_positions:
                pair_positions[pair].discard(token_idx)
                if not pair_positions[pair]:  # Empty set
                    del pair_positions[pair]
        
        for pair in new_pairs:
            pair_counts[pair] += freq
            
            # Update position index: add this token to new pair
            if pair not in pair_positions:
                pair_positions[pair] = set()
            pair_positions[pair].add(token_idx)

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
    tracemalloc.start()

    special_tokens = ["<|endoftext|>"]
    input_path = FIXTURES_PATH + "/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    
    print(f"Training BPE on: '{input_path}' with {vocab_size} vocab_size...")

    profiler = cProfile.Profile()
    profiler.enable()

    result = train_bpe(input_path, vocab_size, special_tokens)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage: {peak / 1024 / 1024 / 1024:.2f} GB")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    # Save to disk
    with open("tinystories_vocab.pkl", "wb") as f:
        pickle.dump(result.vocab, f)

    with open("tinystories_merges.pkl", "wb") as f:
        pickle.dump(result.merges, f)

    print("Saved vocab and merges to disk")

    # Find longest token
    longest_token = max(result.vocab.values(), key=len)
    longest_token_id = [k for k, v in result.vocab.items() if v == longest_token][0]

    print(f"\nLongest token:")
    print(f"  ID: {longest_token_id}")
    print(f"  Length: {len(longest_token)} bytes")
    print(f"  Value: {longest_token}")
    print(f"  As string: {longest_token.decode('utf-8', errors='replace')}")
