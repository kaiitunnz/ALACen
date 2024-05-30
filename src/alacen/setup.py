import subprocess
from pathlib import Path


def setup(conda_env: str):
    def run_shell(command: str) -> int:
        return subprocess.call(f"conda run -n {conda_env} {command}", shell=True)

    if run_shell("conda install -y conda-forge::ffmpeg"):
        raise Exception("Failed to install ffmpeg")

    package_dir = Path(__file__).parent

    # Install dependencies
    dependencies = [
        "git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft",
        "xformers==0.0.22",
        "torchaudio==2.0.2 torch==2.0.1 torchvision==0.15.2",
        "tensorboard==2.16.2",
        "phonemizer==3.2.1",
        "datasets==2.16.0",
        "torchmetrics==0.11.1",
        "huggingface_hub==0.22.2",
    ]
    for d in dependencies:
        if run_shell(f"pip install {d}"):
            raise Exception(f"Failed to install {d}")

    # Install MFA for getting forced-alignment, this could take a few minutes
    if run_shell(
        "conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068",
    ):
        raise Exception("Failed to install MFA")
    # Install MFA english dictionary and model
    if run_shell("mfa model download dictionary english_us_arpa"):
        raise Exception(f"Failed to install MFA dictionary")
    if run_shell("mfa model download acoustic english_us_arpa"):
        raise Exception(f"Failed to install MFA acoustic")

    diff2lip_dir = package_dir / "lipsync/diff2lip"
    # Install Diff2Lip dependencies
    if run_shell(f"pip install {diff2lip_dir / 'guided-diffusion'}"):
        raise Exception("Failed to install guided-diffusion")
    if run_shell("pip install gdown"):
        raise Exception("Failed to install gdown")
    if run_shell(
        f"gdown --folder 1UMiHAhVf5M_CKzjVQFC5jkz-IXAAnFo5 -O {diff2lip_dir / 'Diff2Lip_checkpoints'}",
    ):
        raise Exception("Failed to download Diff2Lip checkpoints")
