import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torchaudio

from .. import TTS
from .data.tokenizer import AudioTokenizer, TextTokenizer
from .models.voicecraft import VoiceCraft
from .inference_tts_scale import inference_one_sample


@dataclass
class MfaAlignmentEntry:
    begin: float
    end: float
    label: str
    type: str
    speaker: str

    @classmethod
    def from_line(cls, line: str):
        begin, end, label, type, speaker = line.strip().split(",")
        return cls(float(begin), float(end), label, type, speaker)


@dataclass
class VoiceCraftArgs:
    out_dir: Union[str, Path]
    tmp_dir: Union[str, Path]
    audio_fname: str
    original_transcript: str
    target_transcript: str
    cut_off_sec: Optional[float] = None
    margin: float = 0.08
    cutoff_tolerance: float = 1
    beam_size: int = 50
    retry_beam_size: int = 200
    stop_repetition: int = 3
    sample_batch_size: int = 3
    codec_audio_sr: int = 16000
    codec_sr: int = 50
    top_k: int = 0
    top_p: float = 0.9
    temperature: float = 1
    silence_tokens: Tuple[int, ...] = (1388, 1898, 131)
    kvcache: float = 1
    seed: int = -1
    verbose: bool = True
    device: str = "cuda"

    @property
    def decode_config(self):
        return {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "stop_repetition": self.stop_repetition,
            "kvcache": self.kvcache,
            "codec_audio_sr": self.codec_audio_sr,
            "codec_sr": self.codec_sr,
            "silence_tokens": self.silence_tokens,
            "sample_batch_size": self.sample_batch_size,
        }


class VoiceCraftTTS(TTS):
    def __init__(self, model_name: Union[str, Path], default_device: Any = "cpu"):
        self.model = VoiceCraft.from_pretrained(
            f"pyp1/VoiceCraft_{model_name.replace('.pth', '')}"
        )
        self.phn2num = self.model.args.phn2num
        self.config = vars(self.model.args)
        self.default_device = default_device

        encodec_fn = Path(__file__).parent / "pretrained_models/encodec_4cb2048_giga.th"
        if not os.path.exists(encodec_fn):
            os.system(
                f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O {encodec_fn}"
            )

        # will also put the neural codec model on gpu
        self.audio_tokenizer = AudioTokenizer(device="cpu", signature=encodec_fn)
        self.text_tokenizer = TextTokenizer(backend="espeak")

    def get_mfa_path(self, out_dir: Union[str, Path], audio_fname: str) -> Path:
        return (Path(out_dir) / "mfa_alignments" / audio_fname).with_suffix(".csv")

    def forced_alignment(
        self,
        out_dir: Union[str, Path],
        audio_fname: str,
        transcript: str,
        beam_size: int = 50,
        retry_beam_size: int = 200,
        ignore_exist: bool = True,
        verbose: bool = True,
    ):
        if ignore_exist and self.get_mfa_path(out_dir, audio_fname).exists():
            return

        out_dir = Path(out_dir)
        align_dir = out_dir / "mfa_alignments"
        with open((out_dir / audio_fname).with_suffix(".txt"), "w") as f:
            f.write(transcript)
        os.system(
            f"mfa align{' -v' if verbose else ' -q'} --clean -j 1 --output_format csv {out_dir} \
            english_us_arpa english_us_arpa {align_dir} --beam {beam_size} --retry_beam {retry_beam_size}"
        )

    def parase_mfa(self, mfa_path: Union[str, Path]) -> List[MfaAlignmentEntry]:
        with open(mfa_path) as f:
            next(f)  # skip the header
            result = [MfaAlignmentEntry.from_line(line) for line in f]
            return [entry for entry in result if entry.type == "words"]

    def to(self, device: Any) -> "VoiceCraftTTS":
        self.model.to(device)
        self.audio_tokenizer.to(device)
        return self

    def generate(self, args: VoiceCraftArgs, auto_mfa: bool = True) -> Path:
        out_dir = args.out_dir
        tmp_dir = args.tmp_dir
        audio_fname = args.audio_fname
        audio_path = Path(tmp_dir) / audio_fname
        audio_file = audio_path.with_suffix(".wav")
        mfa_path = self.get_mfa_path(tmp_dir, args.audio_fname)

        if not auto_mfa:
            if not mfa_path.exists():
                raise ValueError(
                    "MFA alignments not found. Make sure to run forced alignment first."
                )
            if not audio_file.exists():
                raise ValueError(
                    "Audio file not found. Make sure to run forced alignment first."
                )

        self.forced_alignment(
            tmp_dir,
            audio_fname,
            args.original_transcript,
            args.beam_size,
            args.retry_beam_size,
            ignore_exist=True,
            verbose=args.verbose,
        )
        alignments = self.parase_mfa(mfa_path)
        cut_off_sec, cut_off_word_idx = self._find_closest_word_boundary(
            alignments, args.cut_off_sec, args.margin, args.cutoff_tolerance
        )
        target_transcript = (
            " ".join(args.original_transcript.split(" ")[: cut_off_word_idx + 1])
            + " "
            + args.target_transcript
        )

        info = torchaudio.info(audio_file)
        audio_dur = info.num_frames / info.sample_rate
        if cut_off_sec >= audio_dur:
            raise ValueError(
                f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
            )

        prompt_end_frame = int(cut_off_sec * info.sample_rate)

        self.to(args.device)
        _, gen_audio = inference_one_sample(
            self.model,
            Namespace(**self.config),
            self.phn2num,
            self.text_tokenizer,
            self.audio_tokenizer,
            audio_file,
            target_transcript,
            args.device,
            args.decode_config,
            prompt_end_frame,
        )
        self.to(self.default_device)

        gen_audio = gen_audio[0].cpu()
        seg_save_fn_gen = Path(f"{out_dir}/{audio_file.stem}_gen_seed{args.seed}.wav")
        torchaudio.save(seg_save_fn_gen, gen_audio, args.codec_audio_sr)
        return seg_save_fn_gen

    def _find_closest_word_boundary(
        self,
        alignments: List[MfaAlignmentEntry],
        cut_off_sec: Optional[float],
        margin: float,
        cutoff_tolerance: float = 1,
    ) -> Tuple[float, int]:
        if cut_off_sec is None:
            return alignments[-1].end, len(alignments) - 1

        cutoff_time = None
        cutoff_index = None
        cutoff_time_best = None
        cutoff_index_best = None
        for i, entry in enumerate(alignments):
            end = entry.end
            if end >= cut_off_sec and cutoff_time == None:
                cutoff_time = entry.end
                cutoff_index = i
            if (
                end >= cut_off_sec
                and end < cut_off_sec + cutoff_tolerance
                and i < len(alignments) - 1
                and alignments[i + 1].begin - end >= margin
            ):
                cutoff_time_best = end + margin * 2 / 3
                cutoff_index_best = i
                break

        if cutoff_time_best != None:
            cutoff_time = cutoff_time_best
            cutoff_index = cutoff_index_best
        return cutoff_time, cutoff_index
