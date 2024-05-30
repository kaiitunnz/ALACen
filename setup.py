from setuptools import setup  # type:ignore

from src.alacen import __version__

setup(
    name="alacen",
    version=__version__,
    url="https://github.com/kaiitunnz/ALACen",
    author="Noppanat Wadlom",
    author_email="noppanat.wad@gmail.com",
    package_data={"alacen": ["py.typed"]},
    package_dir={"": "src"},
)
