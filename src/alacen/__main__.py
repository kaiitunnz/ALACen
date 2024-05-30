from argparse import ArgumentParser
from pathlib import Path

from .setup import setup


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "action", nargs="?", choices=["setup"], default=None, help="ALACen command"
    )
    parser.add_argument("--video", type=Path, help="Path to the video file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose mode", default=False
    )
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Name of the Conda environment. This is required when running setup.",
    )
    return parser.parse_args()


args = parse_args()

if args.action == "setup":
    if args.conda_env is None:
        raise ValueError("Conda environment name is required for setup")
    setup(args.conda_env)
    exit()

try:
    import torch

    from .alacen import ALACen
    from .asr.whisper import Whisper
    from .paraphrase.pegasus import PegasusAlacen
    from .tts.voicecraft.voicecraft import VoiceCraftTTS, VoiceCraftArgs
    from .lipsync.diff2lip.diff2lip import Diff2Lip, Diff2LipArgs
except ImportError:
    raise ImportError(
        "ALACen is not installed. Please run `python -m alacen setup` to install ALACen."
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

asr = Whisper()
paraphrase = PegasusAlacen()
tts = VoiceCraftTTS(model_name="330M_TTSEnhanced")
lipsync = Diff2Lip(Diff2LipArgs())

alacen = ALACen(asr, paraphrase, tts, lipsync)

alacen.run(
    args.video,
    "output",
    VoiceCraftArgs,
    num_paraphrases=5,
    device=device,
    verbose=args.verbose,
    clean_up=True,
)
