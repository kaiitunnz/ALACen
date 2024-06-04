import atexit
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Union

import ffmpeg

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
        self.logger = self._init_logger()

    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger("alacen")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s | %(name)s | %(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def run(
        self,
        video_path: Union[str, Path],
        out_dir: Union[str, Path],
        tts_args: Callable[..., Any],
        tmp_dir: Optional[Union[str, Path]] = None,
        num_paraphrases: int = 1,
        merge_av: bool = True,
        mode: Literal["auto", "semi"] = "auto",
        verbose: bool = True,
        device: Any = "cuda",
        clean_up: bool = True,
    ):
        self._validate_args(num_paraphrases=num_paraphrases, mode=mode)

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        video_path = Path(video_path)
        if tmp_dir is None:
            tmp_dir = video_path.with_suffix("")
        tmp_dir = Path(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        if clean_up:
            _to_clean_up.append(tmp_dir)

        # Extract Audio
        self.logger.debug("Extracting audio from video...")
        original_audio_path = tmp_dir / video_path.with_suffix(".wav").name
        self.extract_audio(video_path, original_audio_path)

        # Speech Recognition
        self.logger.debug("Performing speech recognition...")
        transcript = self.asr.transcribe(original_audio_path, device)
        self.logger.debug(f"Transcript: {transcript}")

        # Paraphrase Generation
        self.logger.debug("Generating paraphrase...")
        target_transcripts = self.paraphrase.paraphrase(transcript, num_paraphrases)
        if num_paraphrases > 1 and mode == "semi":
            print("Please choose the best paraphrase among the following:")
            while True:
                for i, candidate in enumerate(target_transcripts, 1):
                    print(f"{i}. {candidate.strip()}", flush=True)
                choice_str = input(
                    f"Enter the number of the best paraphrase (1 - {len(target_transcripts)}) or manually enter your own paraphrase ('r' to retry): "
                )
                if choice_str.lower() == "r":
                    target_transcripts = self.paraphrase.paraphrase(
                        transcript, num_paraphrases
                    )
                    continue
                if not choice_str.isdigit():
                    target_transcript = choice_str
                    break
                choice = int(choice_str)
                if not (1 <= choice <= len(target_transcripts)):
                    print("Invalid choice. Please try again.")
                    continue
                target_transcript = target_transcripts[choice - 1]
                break
        else:
            target_transcript = target_transcripts[0]
        print("Selected paraphrase:", target_transcript.strip(), flush=True)

        os.makedirs(out_dir, exist_ok=True)

        # Text-to-Speech
        self.logger.debug("Generating new audio...")
        while True:
            generated_audio_paths = self.tts.generate(
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
            if len(generated_audio_paths) == 1 or mode == "auto":
                generated_audio_path = generated_audio_paths.pop(0)
                print(
                    f"Generated audio file saved to '{generated_audio_path}'",
                    flush=True,
                )
                if mode == "semi":
                    choice = input("Do you want to retry? (y/n): ")
                    if choice.lower() == "y":
                        continue
            else:
                print("Generated audio files saved to:")
                for i, path in enumerate(generated_audio_paths, 1):
                    print(f"  {i}. {path}")
                choice_str = input(
                    f"Enter the number of the best audio (1 - {len(generated_audio_paths)}) ('r' to retry): "
                )
                if choice_str.isdigit():
                    choice = int(choice_str)
                    if choice < 1 or choice > len(generated_audio_paths):
                        print("Invalid choice. Please try again.")
                        continue
                else:
                    if choice_str.lower() != "r":
                        print("Invalid choice. Please try again.")
                    continue
                generated_audio_path = generated_audio_paths.pop(choice - 1)
            _to_clean_up.extend(generated_audio_paths)
            break

        # Lip Synchronization
        out_path = Path(out_dir) / f"{video_path.stem}_censored.mp4"
        self.logger.debug("Generating lip-synced video...")
        self.lipsync.generate(video_path, generated_audio_path, out_path)

        if merge_av:
            self.logger.debug("Merging generated audio and video...")
            merged_path = out_path.with_stem(f"{video_path.stem}_censored_merged")
            self.merge_av(out_path, generated_audio_path, merged_path)
            os.remove(generated_audio_path)
            os.remove(out_path)
            merged_path.rename(out_path)

        if clean_up:
            clean_up_func()

        self.logger.debug("DONE")
        self.logger.setLevel(logging.INFO)

    def extract_audio(self, input_file: Path, output_file: Path):
        process = ffmpeg.input(str(input_file)).output(
            str(output_file), acodec="pcm_s16le", ar="44100", ac=2, loglevel="quiet"
        )
        process.run()

    def merge_av(self, video_path: Path, audio_path: Path, out_path: Path):
        # ffmpeg -i v.mp4 -i a.wav -c:v copy -map 0:v:0 -map 1:a:0 new.mp4 -shortest
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)
        process = ffmpeg.output(
            input_video,
            input_audio,
            str(out_path),
            c="copy",
            shortest=None,
            loglevel="quiet",
        ).global_args("-map", "0:v:0", "-map", "1:a:0")
        process.run()

    def _validate_args(
        self, num_paraphrases: int, mode: Literal["auto", "semi"] = "auto"
    ):
        if num_paraphrases < 1:
            raise ValueError("num_paraphrases must be greater than or equal to 1.")
        if mode not in ["auto", "semi"]:
            raise ValueError("Invalid execution mode.")
