from dataclasses import dataclass

@dataclass(frozen=True)
class BPETokenizerParamsV2:
    vocab: dict[int, bytes]  # {0: b' ', 1: b't', 2: b'he', 3: b'r'}
    merges: list[tuple[bytes, bytes]] # [(b' ', b't'), (b'he', b'r')]