set -e 
set -o pipefail

mfa_path="$HOME/Documents/MFA/pretrained_models"

mamba install -y conda-forge::ffmpeg

pip install numpy
pip install -r requirements.txt
pip install gdown

# Install VoiceCraft
pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
pip install xformers==0.0.22
pip install torchaudio==2.0.2 torch==2.0.1 torchvision==0.15.2

pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2
# Install MFA for getting forced-alignment, this could take a few minutes
mamba install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# Prepare the directory for downloading MFA pre-trained models
timeout 10 mfa model download dictionary english_us_arpa || [[ $? -eq 124 ]]
# Download pre-trained models
gdown 1DczEcOKbi86g18TtUwgtAh91hiG6Bi9y -O ${mfa_path}/dictionary/ # dictionary english_us_arpa
gdown 1Vf330klR0bTWNPQOOcpfMwd_mxG_O68H -O ${mfa_path}/acoustic/ # acoustic english_us_arpa

# Install Diff2Lip
cd src/alacen/lipsync/diff2lip
pip install -e ./guided-diffusion
gdown --folder 1UMiHAhVf5M_CKzjVQFC5jkz-IXAAnFo5
cd -

# For running jupyter notebooks
pip install ipykernel