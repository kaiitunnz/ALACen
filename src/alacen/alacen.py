import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Union

import ffmpeg

from .asr import ASR
from .lipsync import LipSync
from .paraphrase import ParaphraseGenerator
from .session import manager as session_manager
from .tts import TTS


class ALACenSession:
    def __init__(
        self,
        alacen: "ALACen",
        video_path: Union[str, Path],
        tmp_dir: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        merge_av: bool = True,
        clean_up: bool = True,
        verbose: bool = True,
        device: Any = "cuda",
    ):
        self.alacen = alacen
        self.video_path = Path(video_path)
        self.merge_av = merge_av
        self.clean_up = clean_up
        self.verbose = verbose
        self.device = device

        self.id = session_manager.create_session()
        self.logger = self.alacen.logger.getChild("alacen.session")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.tmp_dir = (
            self.video_path.with_suffix("") if tmp_dir is None else Path(tmp_dir)
        )
        os.makedirs(self.tmp_dir, exist_ok=True)
        if self.clean_up:
            session_manager.add_resource(self.id, self.tmp_dir)

        self.out_dir = (
            Path(str(self.tmp_dir) + "_out") if out_dir is None else Path(out_dir)
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.original_audio_path = (
            self.tmp_dir / self.video_path.with_suffix(".wav").name
        )

    def asr(self) -> str:
        # Extract Audio
        self.logger.debug("Extracting audio from video...")
        self.alacen.extract_audio(self.video_path, self.original_audio_path)

        # Speech Recognition
        self.logger.debug("Performing speech recognition...")
        transcript = self.alacen.asr.transcribe(self.original_audio_path, self.device)
        self.logger.debug(f"Transcript: {transcript}")

        return transcript

    def paraphrase(self, transcript: str, num_paraphrases: int = 1) -> List[str]:
        self.logger.debug("Generating paraphrase...")
        return self.alacen.paraphrase.paraphrase(
            transcript, num_paraphrases, device=self.device
        )

    def tts(
        self, transcript: str, target_transcript: str, tts_args: Callable[..., Any]
    ) -> List[Path]:
        self.logger.debug("Generating new audio...")
        generated_audio_paths = self.alacen.tts.generate(
            tts_args(
                out_dir=self.out_dir,
                tmp_dir=self.tmp_dir,
                audio_fname=self.original_audio_path.name,
                original_transcript=transcript,
                target_transcript=target_transcript,
                verbose=self.verbose,
                device=self.device,
            )
        )
        return generated_audio_paths

    def select_tts_audio(self, choice: int, generated_audio_paths: List[Path]) -> Path:
        generated_audio_path = generated_audio_paths.pop(choice)
        session_manager.add_resources(self.id, generated_audio_paths)
        return generated_audio_path

    def lipsync(self, generated_audio_path: str) -> Path:
        out_path = self.out_dir / f"{self.video_path.stem}_censored.mp4"
        self.logger.debug("Generating lip-synced video...")
        self.alacen.lipsync.generate(
            str(self.video_path), generated_audio_path, out_path
        )

        if self.merge_av:
            self.logger.debug("Merging generated audio and video...")
            merged_path = out_path.with_stem(f"{self.video_path.stem}_censored_merged")
            self.alacen.merge_av(out_path, generated_audio_path, merged_path)
            os.remove(out_path)
            merged_path.rename(out_path)

        return out_path

    def __del__(self):
        if self.clean_up:
            session_manager.remove_session(self.id)


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

    def create_session(
        self,
        video_path: Union[str, Path],
        tmp_dir: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        merge_av: bool = True,
        verbose: bool = True,
        device: Any = "cuda",
        clean_up: bool = True,
    ) -> ALACenSession:
        return ALACenSession(
            alacen=self,
            video_path=video_path,
            tmp_dir=tmp_dir,
            out_dir=out_dir,
            merge_av=merge_av,
            clean_up=clean_up,
            verbose=verbose,
            device=device,
        )

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

        session = self.create_session(
            video_path=video_path,
            tmp_dir=tmp_dir,
            out_dir=out_dir,
            merge_av=merge_av,
            verbose=verbose,
            device=device,
            clean_up=clean_up,
        )

        # Speech Recognition
        transcript = session.asr()

        # Paraphrase Generation
        target_transcripts = session.paraphrase(transcript, num_paraphrases)
        if num_paraphrases > 1 and mode == "semi":
            print("Please choose the best paraphrase among the following:")
            while True:
                for i, candidate in enumerate(target_transcripts, 1):
                    print(f"{i}. {candidate.strip()}", flush=True)
                choice_str = input(
                    f"Enter the number of the best paraphrase (1 - {len(target_transcripts)}) or manually enter your own paraphrase ('r' to retry): "
                )
                if choice_str.lower() == "r":
                    target_transcripts = session.paraphrase(transcript, num_paraphrases)
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
        while True:
            generated_audio_paths = session.tts(transcript, target_transcript, tts_args)
            if len(generated_audio_paths) == 1 or mode == "auto":
                generated_audio_path = session.select_tts_audio(
                    0, generated_audio_paths
                )
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
                generated_audio_path = session.select_tts_audio(
                    choice - 1, generated_audio_paths
                )
            break

        # Lip Synchronization
        session.lipsync(generated_audio_path)

        self.logger.debug("DONE")

    def extract_audio(self, input_file: Path, output_file: Path):
        process = ffmpeg.input(str(input_file)).output(
            str(output_file), acodec="pcm_s16le", ar="44100", ac=2, loglevel="quiet"
        )
        process = process.global_args("-nostdin")
        process.run(overwrite_output=True)

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
        process = process.global_args("-nostdin")
        process.run(overwrite_output=True)

    def _validate_args(
        self, num_paraphrases: int, mode: Literal["auto", "semi"] = "auto"
    ):
        if num_paraphrases < 1:
            raise ValueError("num_paraphrases must be greater than or equal to 1.")
        if mode not in ["auto", "semi"]:
            raise ValueError("Invalid execution mode.")
