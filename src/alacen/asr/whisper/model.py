from pathlib import Path
from typing import Any, Union

import whisper

from .. import ASR


class Whisper(ASR):
    def __init__(self, model_name: str = "small.en", default_device: Any = "cpu"):
        self.model = whisper.load_model(model_name, device=default_device)
        self.default_device = default_device

    def transcribe(self, audio_path: Union[str, Path], device: Any = "cuda") -> str:
        audio = whisper.load_audio(audio_path)
        self.model.to(device)
        result = self.model.transcribe(audio)
        self.model.to(self.default_device)
        return result["text"]
