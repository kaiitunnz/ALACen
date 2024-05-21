from argparse import ArgumentParser
from pathlib import Path

import torch

from .alacen import ALACen
from .asr.whisper import Whisper
from .paraphrase.gemini import GeminiClient
from .tts.voicecraft.voicecraft import VoiceCraftTTS, VoiceCraftArgs
from .lipsync.diff2lip.diff2lip import Diff2Lip, Diff2LipArgs


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--video", type=Path, help="Path to the video file")
    return parser.parse_args()


args = parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

asr = Whisper()
paraphrase = GeminiClient()
tts = VoiceCraftTTS(model_name="330M_TTSEnhanced")
lipsync = Diff2Lip(Diff2LipArgs())

alacen = ALACen(asr, paraphrase, tts, lipsync)

alacen.run(
    args.video,
    "output",
    VoiceCraftArgs,
    num_paraphrases=5,
    device=device,
    verbose=False,
    clean_up=True,
)
