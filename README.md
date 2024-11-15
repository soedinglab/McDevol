# McDevol
A metagenome binning tool based on semi-contrastive learning method using the framework of BYOL (Bootstrap Your Own Latent) model. It only requires positive augmentated pairs for contrastive learning. As input, it integrates k-mer sequence embedding from GenomeFace and sampled contig coverage profile using binomial sampling of augmented pairs for training.

[Mcdevol_byol_model.pdf](https://github.com/user-attachments/files/17779212/Mcdevol_byol_model.pdf)

# Installation
    CONDA_OVERRIDE_GLIBC=2.17 CONDA_OVERRIDE_CUDA=10.2 conda env create --file=environment.yml
    export PATH=${PATH}/$(pwd)
    python mcdevol/mcdevol.py --help

Require glibc2.25

