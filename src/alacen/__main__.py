from argparse import ArgumentParser
from pathlib import Path

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--video", type=Path, help="Path to the video file")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output directory", default=Path("output")
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose mode", default=False
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["auto", "semi"],
        help="ALACen's mode of operation",
        default="auto",
    )
    parser.add_argument(
        "-np",
        "--num-paraphrases",
        type=int,
        help="Number of candidate paraphrases",
        default=5,
    )
    parser.add_argument(
        "-ns",
        "--num-speeches",
        type=int,
        help="Number of candidate speeches",
        default=5,
    )
    parser.add_argument("-d", "--device", help="Device to run on", default=DEVICE)
    parser.add_argument(
        "--num-gpus", type=int, help="Number of GPUs to run on", default=3
    )
    return parser.parse_args()


args = parse_args()

try:
    from .alacen import ALACen
    from .asr.whisper import Whisper
    from .paraphrase.pegasus import PegasusAlacen
    from .tts.voicecraft.voicecraft import VoiceCraftTTS, VoiceCraftArgs
    from .lipsync.diff2lip.diff2lip import Diff2Lip, Diff2LipArgs
except ImportError:
    raise ImportError(
        "ALACen has not been set up. Please run `bash setup.sh` to set up ALACen."
    )


asr = Whisper()
paraphrase = PegasusAlacen()
tts = VoiceCraftTTS(model_name="330M_TTSEnhanced")
voicecraft_args = VoiceCraftArgs.constructor(num_samples=args.num_speeches)
lipsync = Diff2Lip(Diff2LipArgs(num_gpus=args.num_gpus))

alacen = ALACen(asr, paraphrase, tts, lipsync)

alacen.run(
    args.video,
    args.output,
    voicecraft_args,
    num_paraphrases=args.num_paraphrases,
    device=args.device,
    mode=args.mode,
    verbose=args.verbose,
    clean_up=True,
)
