import subprocess
from pathlib import Path


def setup():
    if subprocess.call("conda install -y conda-forge::ffmpeg", shell=True):
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
        if subprocess.call(f"pip install {d}", shell=True):
            raise Exception(f"Failed to install {d}")

    # Install MFA for getting forced-alignment, this could take a few minutes
    if subprocess.call(
        "conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068",
        shell=True,
    ):
        raise Exception("Failed to install MFA")
    # Install MFA english dictionary and model
    if subprocess.call("mfa model download dictionary english_us_arpa", shell=True):
        raise Exception(f"Failed to install MFA dictionary")
    if subprocess.call("mfa model download acoustic english_us_arpa", shell=True):
        raise Exception(f"Failed to install MFA acoustic")

    diff2lip_dir = package_dir / "src/alacen/lipsync/diff2lip"
    # Install Diff2Lip dependencies
    if subprocess.call(
        f"pip install -e {diff2lip_dir / 'guided-diffusion'}", shell=True
    ):
        raise Exception("Failed to install guided-diffusion")
    if subprocess.call(
        f"pip install gdown && gdown --folder 1UMiHAhVf5M_CKzjVQFC5jkz-IXAAnFo5 -O {diff2lip_dir / 'Diff2Lip_checkpoints'}",
        shell=True,
    ):
        raise Exception("Failed to download Diff2Lip checkpoints")
