from abc import abstractmethod
from typing import List


class ParaphraseGenerator:
    @abstractmethod
    def paraphrase(self, speech: str, n: int = 1, **kwargs) -> List[str]:
        raise NotImplementedError()
