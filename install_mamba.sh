set -e 
set -o pipefail

pip install gdown

# Install mamba
gdown 1RPehNyeJcPf-KPtYqLt9SfoHeBapnlr0
bash Mambaforge-24.3.0-0-Linux-x86_64.sh
rm Mambaforge-24.3.0-0-Linux-x86_64.sh
