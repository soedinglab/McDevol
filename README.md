# McDevol
# Installation
    CONDA_OVERRIDE_GLIBC=2.17 CONDA_OVERRIDE_CUDA=10.2 conda env create --file=environment.yml
    export PATH=${PATH}/$(pwd)
    python mcdevol/mcdevol.py --help

Require glibc2.25
