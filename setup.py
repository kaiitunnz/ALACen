import subprocess
from setuptools import setup  # type:ignore

from src.alacen import __version__

with open("requirements.txt", "r") as f:
    install_requires = [line for line in f if not line.lstrip().startswith("#")]

if subprocess.call("pip install numpy", shell=True):
    raise Exception("Failed to install numpy")

setup(
    name="alacen",
    version=__version__,
    url="https://github.com/kaiitunnz/ALACen",
    author="Noppanat Wadlom",
    author_email="noppanat.wad@gmail.com",
    package_data={"alacen": ["py.typed"]},
    package_dir={"": "src"},
    install_requires=install_requires,
)
