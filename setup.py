from setuptools import find_packages, setup

setup(
    name="dainet",
    version="0.1.0",
    description="DAINet - Direction-Aware Illumination Network: single-image illumination normalization.",
    author="Anirban Das",
    packages=find_packages(exclude=("tests", "tests.*", "scripts", "scripts.*", "docs")),
    python_requires=">=3.10",
)
