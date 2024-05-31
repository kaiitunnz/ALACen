# ALACen: Automatic Language-level Adjustment for Video Censorship

## Installation

This installation guide assumes that you already have Conda installed. If not so, follow the Conda installation guide [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) before you proceed.

1. Create a new Conda environment with Python 3.9 and activate it. For example,

```
conda create -n alacen python=3.9 && conda activate alacen
```

2. Install Mamba using the following command. You need to restart your terminal after the installation finishes. We need Mamba because installing the dependencies with conda through the conda-forge channel often hangs.
   _Note: You can skip this step if you already have Mamba installed._

```
bash install_mamba.sh
```

3. Run the following command to install the dependencies and download pre-trained models. If it fails with a connection error, try running it again.

```
bash setup.sh
```

## How to run

We provide two options for running ALACen.

1. Execute the Python module.

```
python -m src.alacen --video <path-to-your-video>
```

2. Run the `run.ipynb` file. This gives you an interactive execution of ALACen. Put your configuration parameters in the Configuration cell and run all the cells. If you encounter the prompt saying files already exist, try removing those files and rerun the cell.
