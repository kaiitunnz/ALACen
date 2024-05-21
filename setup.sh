pip install -r requirements.txt

conda install -y conda-forge::ffmpeg

# Install VoiceCraft
pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
pip install xformers==0.0.22
pip install torchaudio==2.0.2 torch==2.0.1 torchvision==0.15.2

pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2
# install MFA for getting forced-alignment, this could take a few minutes
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# install MFA english dictionary and model
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa

# Install Diff2Lip
cd src/alacen/lipsync/diff2lip
pip install -e ./guided-diffusion
pip install gdown
gdown --folder 1UMiHAhVf5M_CKzjVQFC5jkz-IXAAnFo5
cd -