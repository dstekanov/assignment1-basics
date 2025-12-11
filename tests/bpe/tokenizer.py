from collections.abc import Iterable, Iterator
import re
import regex
import json
from functools import lru_cache

@lru_cache
def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # Create reverse mapping: bytes -> id for fast lookup
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        # Track next available ID for merges (optimization to avoid max() calls)
        self.next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        
        # GPT-2 pre-tokenization pattern
        self._pretokenize_pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        # Build special token pattern for splitting
        if self.special_tokens:
            # Sort by length (longest first) to handle overlapping special tokens
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(token) for token in sorted_special)
            self._special_token_pattern = re.compile(pattern)
        else:
            self._special_token_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens."""
        # Get GPT-2 byte decoder
        gpt2_byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}
        
        # Load vocab from JSON file: {"token_string": id, ...}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            gpt2_vocab = json.load(f)
        
        # Convert vocab from GPT-2 format to {id: bytes}
        # Each token_str in vocab is a sequence of GPT-2 encoded characters
        vocab = {}
        for token_str, token_id in gpt2_vocab.items():
            # Decode each character in token_str using gpt2_byte_decoder
            token_bytes = bytes([gpt2_byte_decoder[char] for char in token_str])
            vocab[token_id] = token_bytes
        
        # Load merges from text file: each line is "token1 token2"
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:  # Skip empty lines
                    continue
                parts = line.split(' ')
                if len(parts) == 2:
                    token1_str, token2_str = parts
                    # Decode token strings using gpt2_byte_decoder
                    token1_bytes = bytes([gpt2_byte_decoder[char] for char in token1_str])
                    token2_bytes = bytes([gpt2_byte_decoder[char] for char in token2_str])
                    merges.append((token1_bytes, token2_bytes))
        
        # If any of the special tokens don't exist in the vocab, append them
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
        
        return cls(vocab, merges, special_tokens)

    def encode(self, string: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # Step 1: Handle special tokens by splitting
        if self._special_token_pattern:
            parts = self._special_token_pattern.split(string)
            matches = self._special_token_pattern.findall(string)
            
            indicies = []
            for i, part in enumerate(parts):
                if part:  # Encode non-special part
                    indicies.extend(self._encode_text(part))
                if i < len(matches):  # Add special token ID
                    special_token = matches[i]
                    special_bytes = special_token.encode("utf-8")
                    special_id = self.bytes_to_id.get(special_bytes)
                    if special_id is not None:
                        indicies.append(special_id)
        else:
            indicies = self._encode_text(string)
        
        return indicies
    
    def _encode_text(self, text: str) -> list[int]:
        """Encode text by pre-tokenizing and applying merges."""
        # Step 1: Pre-tokenize using GPT-2 pattern
        pretokens = self._pretokenize_pattern.findall(text)
        
        indicies = []
        for pretoken in pretokens:
            # Step 2: For each pre-token, convert to bytes and apply merges
            token_ids = self._encode_pretoken(pretoken)
            indicies.extend(token_ids)
        
        return indicies
    
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """Encode a single pre-token by converting to bytes and applying merges."""
        # Convert pre-token string to bytes
        byte_values = pretoken.encode("utf-8")
        # Start with individual byte IDs
        indicies = [self.bytes_to_id[bytes([b])] for b in byte_values]
        
        # Create local copies so we don't modify self.vocab during encoding
        vocab_copy = self.vocab.copy()
        bytes_to_id_copy = self.bytes_to_id.copy()
        
        # Apply merges in order
        indicies = self._merge_pair_if_present(indicies, vocab_copy, bytes_to_id_copy)
        
        # Convert final byte sequences to IDs
        final_ids = []
        for idx in indicies:
            # idx might be an ID from vocab_copy (after merges)
            if idx in vocab_copy:
                token_bytes = vocab_copy[idx]
                # Find the ID for this bytes in self.bytes_to_id (original vocab)
                final_id = self.bytes_to_id.get(token_bytes)
                if final_id is not None:
                    final_ids.append(final_id)
        
        return final_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""
        for line in iterable:
            ids = self.encode(line)
            for id in ids:
                yield id

    def _merge_pair_if_present(self, indicies, vocab, bytes_to_id):
        """Helper method that applies merges using provided vocab and bytes_to_id maps.
        
        Optimizations:
        1. Early exit when sequence is too short
        2. Build set of current pairs once for O(1) lookup
        3. Only iterate through applicable merges
        4. Stop when no more merges can be applied
        """
        # Early exit for short sequences
        if len(indicies) <= 1:
            return indicies
        
        # Track next_id locally to avoid max() calls (huge optimization!)
        next_id = max(bytes_to_id.values()) + 1 if bytes_to_id else 0
        
        # Build set of current pairs for O(1) lookup (Optimization 2)
        def get_pairs_set(ids):
            """Build set of adjacent pairs in O(n) time."""
            if len(ids) < 2:
                return set()
            return set((ids[i], ids[i+1]) for i in range(len(ids) - 1))
        
        current_pairs = get_pairs_set(indicies)
        
        for pair_bytes in self.merges:
            # Early exit: if no pairs left, we're done (Optimization 4)
            if not current_pairs:
                break
            
            # pair_bytes is (bytes, bytes), convert to (int, int)
            if pair_bytes[0] not in bytes_to_id or pair_bytes[1] not in bytes_to_id:
                # Skip merges where one of the tokens doesn't exist yet
                continue
            
            pair_ids = (bytes_to_id[pair_bytes[0]], bytes_to_id[pair_bytes[1]])
            
            # Check if this pair exists in current pairs (O(1) lookup)
            if pair_ids not in current_pairs:
                continue
            
            # Apply merge
            new_indicies = self.merge(indicies, pair_ids, next_id)
            
            # Only update if merge actually happened
            if new_indicies != indicies:
                indicies = new_indicies
                # Update vocab and bytes_to_id for next merge
                merged_bytes = pair_bytes[0] + pair_bytes[1]
                vocab[next_id] = merged_bytes
                bytes_to_id[merged_bytes] = next_id
                next_id += 1
                
                # Rebuild pairs set after merge (necessary for correctness)
                current_pairs = get_pairs_set(indicies)
        
        return indicies

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        # Use cached vocab from last encode if available, otherwise use self.vocab
        vocab = getattr(self, '_vocab_cache', self.vocab)
        # Look up each ID in vocab, replacing missing IDs with empty bytes
        bytes_list = [vocab.get(id, b'') for id in ids]
        # Concatenate all bytes and decode to UTF-8, replacing invalid sequences with U+FFFD
        string = b"".join(bytes_list).decode("utf-8", errors="replace")
        return string

    def merge(self, indicies: list[int], pair: tuple[int, int], new_index: int) -> list[int]:

        new_indicies = []

        i = 0
        while i < len(indicies):
            if i+1 < len(indicies) and indicies[i] == pair[0] and indicies[i+1] == pair[1]:
                new_indicies.append(new_index)
                i += 2
            else:
                new_indicies.append(indicies[i])
                i += 1

        return new_indicies



if __name__ == "__main__":
    string = "abc frd"
    bytes = string.encode("utf-8")
    print("Indicies before encoding: ", bytes)
    