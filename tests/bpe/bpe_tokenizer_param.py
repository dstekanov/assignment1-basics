from dataclasses import dataclass

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[int, int], int] # index1, index2 => merged index