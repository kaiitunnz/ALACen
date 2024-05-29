from pathlib import Path
from typing import Any, List, Optional, Union

import gdown
from transformers import AutoModelForSeq2SeqLM, PegasusTokenizer, GenerationConfig

from . import ParaphraseGenerator
from .utils import count_syllables


class PegasusAlacen(ParaphraseGenerator):
    MODEL_NAME = "pegasus_alacen"
    MODEL_URL = "1HMPHd7cNrNYAySZPlbBzRhtI7Z8pxnXk"

    def __init__(
        self, model_path: Optional[Union[str, Path]] = None, device: Any = "cpu"
    ):
        self.model_path = (
            Path(__file__).parent / self.MODEL_NAME
            if model_path is None
            else model_path
        )
        if not (self.model_path.exists() and self.model_path.is_dir()):
            self.download_model()
        self.default_device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(device)
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_path)

    def paraphrase(
        self,
        speech: str,
        n: int = 1,
        device: Any = "cuda",
        max_new_tokens: int = 100,
        temperature: float = 0.5,
        forced_eos_token_id: Optional[Union[int, List[int]]] = 1,
    ) -> List[str]:
        inputs = self.tokenizer([speech], return_tensors="pt").input_ids
        generation_config = GenerationConfig(
            do_sample=True,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            forced_eos_token_id=forced_eos_token_id,
        )

        self.model.to(device)
        inputs = inputs.to(device)
        outputs = self.model.generate(inputs, generation_config=generation_config)
        self.model.to(self.default_device)

        decoded = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        speech_syllables = count_syllables(speech)
        decoded = sorted(
            decoded, key=lambda x: abs(speech_syllables - count_syllables(x))
        )
        return decoded

    def download_model(self):
        import os
        import shutil

        download_dir = self.model_path.parent
        download_path = self.model_path.with_suffix(".zip")
        gdown.download(id=self.MODEL_URL, output=str(download_path))
        shutil.unpack_archive(download_path, download_dir)
        os.remove(download_path)
