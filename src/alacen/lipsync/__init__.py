from abc import abstractmethod
from pathlib import Path


class LipSync:
    @abstractmethod
    def generate(self, video_path: str, audio_path: str, out_path: str):
        raise NotImplementedError()
