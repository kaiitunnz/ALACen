from abc import abstractmethod
from pathlib import Path
from typing import Any, List


class TTS:
    @abstractmethod
    def generate(self, args: Any, *_) -> List[Path]:
        raise NotImplementedError()
