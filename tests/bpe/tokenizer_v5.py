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
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = re.compile(PAT)
SPECIAL_TOKENS_PATTERN = None

class ReverseBytes:
    """Wrapper for reverse byte comparison in heaps."""
    def __init__(self, b):
        self.b = b
    def __lt__(self, other):
        return self.b > other.b
    def __le__(self, other):
        return self.b >= other.b
    def __gt__(self, other):
        return self.b < other.b
    def __ge__(self, other):
        return self.b <= other.b
    def __eq__(self, other):
        return self.b == other.b
    def __ne__(self, other):
        return self.b != other.b

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

    num_merges = vocab_size - len(vocab)
    print("Number of merging: ", num_merges)

    # Build initial pair counts and position index (cache)
    pair_counts, pair_positions = build_pair_index(token_indicies, freqs)
    
    # Build max heap for fast best pair selection (rustbpe style)
    # Heap stores: (-count, negated_merged_bytes, counter, pair) for max heap
    # Note: Python's heapq is min heap, so negate count and merged_bytes
    # Tie-breaking: when counts equal, use lexicographic order of merged bytes (greater first)
    # negate_bytes() inverts byte values so greater bytes sort first
    pair_heap = []
    counter = 0
    for pair, count in pair_counts.items():
        if count > 0:
            # Use tuple of bytes for tie-breaking (matches reference implementation)
            merged_bytes_tuple = (vocab[pair[0]], vocab[pair[1]])
            # Use ReverseBytes for max heap behavior (greater tuples first)
            heapq.heappush(pair_heap, (-count, ReverseBytes(merged_bytes_tuple), counter, pair))
            counter += 1
    
    print(f"Built heap with {len(pair_heap)} pairs")
    
    for i in range(num_merges):
        # Get best pair from heap with lazy refresh (rustbpe style)
        best_pair, best_count = get_best_pair_from_heap(pair_counts, pair_heap, vocab)
        
        new_id = next_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_id
        next_id += 1

        # Update pair counts incrementally using position index
        # Also push updated pairs back to heap
        counter = update_pair_counts_with_heap(token_indicies, freqs, pair_counts, pair_positions, best_pair, new_id, pair_heap, vocab, counter)
        
        # Periodic heap cleanup to prevent memory bloat (every 1000 merges)
        heap_cleanup(pair_counts, pair_heap, i, vocab)

    merges_list = [(vocab[pair[0]], vocab[pair[1]]) for pair, _ in merges.items()]

    return BPETokenizerParamsV2(vocab, merges_list)

def heap_cleanup(pair_counts, pair_heap, i, vocab):
    if (i + 1) % 1000 == 0 and len(pair_heap) > len(pair_counts) * 3:
            # Rebuild heap from current pair_counts to remove stale entries
        pair_heap.clear()
        counter = 0
        for pair, count in pair_counts.items():
            if count > 0:
                merged_bytes_tuple = (vocab[pair[0]], vocab[pair[1]])
                heapq.heappush(pair_heap, (-count, ReverseBytes(merged_bytes_tuple), counter, pair))
                counter += 1
        print(f"Heap cleanup at merge {i+1}: {len(pair_heap)} valid pairs")

def get_best_pair_from_heap(pair_counts, pair_heap, vocab):
    """
    Get best pair from heap with lazy refresh (rustbpe style).
    
    Lazy refresh: if heap top's count doesn't match current count,
    update and re-push instead of maintaining heap after every update.
    
    Tie-breaking: when counts equal, use lexicographic order of merged bytes.
    
    Returns:
        (best_pair, count): The selected pair and its frequency
    """
    while pair_heap:
        neg_count, reverse_merged_bytes, counter, pair = heapq.heappop(pair_heap)
        current_count = pair_counts.get(pair, 0)
        
        # Lazy refresh: if count changed, update and re-push
        if current_count != -neg_count:
            if current_count > 0:
                current_merged_bytes_tuple = (vocab[pair[0]], vocab[pair[1]])
                heapq.heappush(pair_heap, (-current_count, ReverseBytes(current_merged_bytes_tuple), counter, pair))
            continue
        
        # Found valid pair
        if current_count > 0:
            return pair, current_count
    
    raise ValueError("No valid pairs in heap")

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

def update_pair_counts_with_heap(
    token_indicies: List[List[int]],
    freqs: List[int],
    pair_counts: Counter,
    pair_positions: Dict[Tuple[int, int], set],
    merged_pair: Tuple[int, int],
    new_id: int,
    pair_heap: list,
    vocab: Dict[int, bytes],
    counter: int
):
    """
    Incrementally update pair counts after a merge using position index.
    Also push updated pairs to heap (rustbpe style).
    """
    # Get affected tokens directly from index (O(1)!)
    if merged_pair not in pair_positions:
        return counter
    
    affected_tokens = list(pair_positions[merged_pair])  # Copy to avoid modification during iteration
    
    # Remove merged pair from index
    del pair_positions[merged_pair]
    
    # Track which pairs were updated (to push to heap)
    updated_pairs = set()
    
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
            else:
                updated_pairs.add(pair)
            
            # Update position index: remove this token from old pair
            if pair in pair_positions:
                pair_positions[pair].discard(token_idx)
                if not pair_positions[pair]:  # Empty set
                    del pair_positions[pair]
        
        for pair in new_pairs:
            pair_counts[pair] += freq
            updated_pairs.add(pair)
            
            # Update position index: add this token to new pair
            if pair not in pair_positions:
                pair_positions[pair] = set()
            pair_positions[pair].add(token_idx)
    
    # Push updated pairs to heap (lazy update will handle stale entries)
    # Use counter to maintain insertion order for tie-breaking
    for pair in updated_pairs:
        if pair in pair_counts and pair_counts[pair] > 0:
            merged_bytes_tuple = (vocab[pair[0]], vocab[pair[1]])
            heapq.heappush(pair_heap, (-pair_counts[pair], ReverseBytes(merged_bytes_tuple), counter, pair))
            counter += 1
    
    return counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    max_chunk_size: int = 1024 * 1024 * 1024,  # 1 GB
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

    input_path = FIXTURES_PATH + "/owt_valid.txt"
    vocab_size = 32000  # Reduced to avoid OOM on large dataset    
    
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
    with open("owt_valid_vocab.pkl", "wb") as f:
        pickle.dump(result.vocab, f)

    with open("owt_valid_merges.pkl", "wb") as f:
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
