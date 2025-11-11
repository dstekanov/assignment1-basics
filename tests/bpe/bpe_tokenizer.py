from bpe_tokenizer_param import BPETokenizerParams

class BPETokenizer:
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indicies = list(map(int, string.encode("utf-8")))
        print("Indicies before encoding: ", indicies)
        # Note: this is a very slow implementation
        indicies = self.merge_pair_if_present(indicies)
        print("Indicies after encoding: ", indicies)
        return indicies

    def merge_pair_if_present(self, indicies):
        for pair, new_index in self.params.merges.items():
            print("Pair: ", pair)
            print("New indicies: ", new_index)
            indicies = self.merge(indicies, pair, new_index)
        return indicies

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        print("Byte list decoded: ", bytes_list)
        string = b"".join(bytes_list).decode("utf-8")
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