# McDevol
A metagenome binning tool based on semi-contrastive learning method. It integrates k-mer sequence embedding from GenomeFace and abundance data augmentation using binomial sampling for training.

![McDevol_BYOL_model](https://github.com/user-attachments/assets/8d2c9719-7208-447f-a221-9ca53a549572)

# Installation
    CONDA_OVERRIDE_GLIBC=2.17 CONDA_OVERRIDE_CUDA=10.2 conda env create --file=environment.yml
    export PATH=${PATH}/$(pwd)
    python mcdevol/mcdevol.py --help

Require glibc2.25

