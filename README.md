# McDevol
A metagenome binning tool based on semi-contrastive learning method using the framework of BYOL (Bootstrap Your Own Latent) model. It only requires positive augmentated pairs for contrastive learning. As input, it integrates k-mer sequence embedding from GenomeFace and sampled contig coverage profile using binomial sampling of augmented pairs for training.

![Mcdevol_byol_model](https://github.com/user-attachments/assets/914fa48e-7780-4f86-9747-4df132635045)

# Installation
    CONDA_OVERRIDE_GLIBC=2.17 CONDA_OVERRIDE_CUDA=10.2 conda env create --file=environment.yml
    export PATH=${PATH}/$(pwd)
    python mcdevol/mcdevol.py --help

Require glibc2.25

