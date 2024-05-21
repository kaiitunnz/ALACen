from abc import abstractmethod
from pathlib import Path
from typing import Any


class TTS:
    @abstractmethod
    def generate(self, args: Any, *_) -> Path:
        raise NotImplementedError()
