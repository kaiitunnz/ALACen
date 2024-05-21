from abc import abstractmethod
from typing import Any


class ASR:
    @abstractmethod
    def transcribe(self, audio_path: str, device: Any = "cuda") -> str:
        raise NotImplementedError()
