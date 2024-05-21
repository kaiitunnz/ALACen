import atexit
import os
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import ffmpeg
import torch

from .asr import ASR
from .lipsync import LipSync
from .paraphrase import ParaphraseGenerator
from .tts import TTS


_to_clean_up: List[Path] = []


@atexit.register
def clean_up_func():
    while _to_clean_up:
        file = _to_clean_up.pop()
        if file.is_dir():
            shutil.rmtree(file)
        else:
            os.remove(file)


class ALACen:
    def __init__(
        self, asr: ASR, paraphrase: ParaphraseGenerator, tts: TTS, lipsync: LipSync
    ):
        self.asr = asr
        self.paraphrase = paraphrase
        self.tts = tts
        self.lipsync = lipsync

    def run(
        self,
        video_path: Union[str, Path],
        out_dir: Union[str, Path],
        tts_args: Callable[..., Any],
        tmp_dir: Optional[Union[str, Path]] = None,
        num_paraphrases: int = 1,
        verbose: bool = True,
        device: Any = "cuda",
        clean_up: bool = True,
    ):
        video_path = Path(video_path)
        if tmp_dir is None:
            tmp_dir = video_path.with_suffix("")
        tmp_dir = Path(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        if clean_up:
            _to_clean_up.append(tmp_dir)

        original_audio_path = tmp_dir / video_path.with_suffix(".wav").name
        self.extract_audio(video_path, original_audio_path)

        # Speech Recognition
        transcript = self.asr.transcribe(original_audio_path, device)

        # Paraphrase Generation
        target_transcripts = self.paraphrase.paraphrase(transcript, num_paraphrases)
        if num_paraphrases > 1:
            print("Please choose the best paraphrase among the following:")
            for i, candidate in enumerate(target_transcripts, 1):
                print(f"{i}. {candidate.strip()}")
            choice = int(
                input(
                    f"Enter the number of the best paraphrase (1 - {len(target_transcripts)}): "
                )
            )
            if not (1 <= choice <= len(target_transcripts)):
                raise ValueError("Invalid choice.")
            target_transcript = target_transcripts[choice - 1]
            print("Selected paraphrase:", target_transcript.strip())
        else:
            target_transcript = target_transcripts[0]

        os.makedirs(out_dir, exist_ok=True)

        # Text-to-Speech
        generated_audio_path = self.tts.generate(
            tts_args(
                out_dir=out_dir,
                tmp_dir=tmp_dir,
                audio_fname=original_audio_path.name,
                original_transcript=transcript,
                target_transcript=target_transcript,
                verbose=verbose,
                device=device,
            )
        )

        # Lip Synchronization
        self.lipsync.generate(
            video_path,
            generated_audio_path,
            Path(out_dir) / f"{video_path.stem}_censored.mp4",
        )

        if clean_up:
            clean_up_func()

    def extract_audio(self, input_file: Path, output_file: Path):
        process = ffmpeg.input(str(input_file)).output(
            str(output_file), acodec="pcm_s16le", ar="44100", ac=2, loglevel="quiet"
        )
        process.run()
