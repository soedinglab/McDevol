# McDevol
A new self-supervised metagenome binning tool based on semi-contrastive learning method using the framework of BYOL (Bootstrap Your Own Latent) model. It only requires positive augmentated pairs for contrastive learning and at least 2x faster than COMEBin. It integrates k-mer sequence embedding vectors from GenomeFace composition model and sampled read coverage profiles using multinomial sampling of contigs to learn embedding space. It applies Leiden community detection algorithm to cluster contigs in the embedding space and obtain metagenomic bins.

![Mcdevol_byol_model](https://github.com/user-attachments/assets/28ac2eb9-6545-4824-aa07-6d98b33d9ac0)

# Installation
    conda env create -n mcdevol_env --file=environment.yml
    conda activate mcdevol_env
    export PATH=${PATH}/$(pwd)/mcdevol
    python mcdevol.py --help

McDevol requires glibc2.25. Currently, McDevol was tested only on Linux system. On cluster node, if the default command doesn't work, try creating conda environment using `CONDA_OVERRIDE_GLIBC=2.17 CONDA_OVERRIDE_CUDA=10.2 conda env create --file=environment.yml`.

# Usage
```
usage: mcdevol.py [-h] (-a ABUNDANCE | -i INPUTDIR) -c CONTIGS [-l MINLENGTH] [-o OUTDIR] [-n NCORES] [--abundformat ABUNDFORMAT] [-v] [-f NFRAGMENTS] [--multi_split]                      
                                                                                                                                                                                         
McDevol: An accurate metagenome binning of contigs based on decovolution of abundance and k-mer embedding                                                                                
                                                                                                                                                                                         
optional arguments:                                                                                                                                                                      
  -h, --help            show this help message and exit                                                                                                                                  
  -a ABUNDANCE, --abundance ABUNDANCE                                                                                                                                                    
                        abundance file in TSV format separated by tabs                                                                                                                   
  -i INPUTDIR, --inputdir INPUTDIR                                                                                                                                                       
                        directory that contains SAM files                                                                                                                                
  -c CONTIGS, --contigs CONTIGS                                                                                                                                                          
                        contigs fasta (or zip)                                                                                                                                           
  -l MINLENGTH, --minlength MINLENGTH                                                                                                                                                    
                        minimum length of contigs to be considered for binning
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -n NCORES, --ncores NCORES
                        Number of cores to use
  --abundformat ABUNDFORMAT
                        Format of abundance ('std|metabat', default='std') std:[contigname, s1meancov, s2meancov,...,sNmeancov]; metabat:[contigName, contigLen, totalAvgDepth,
                        s1meancov, s1varcov,...,sNmeancov, sNvarcov]
  -v, --version         print version and exit
  -f NFRAGMENTS, --nfragments NFRAGMENTS
                        number of augumented fragments to generate
  --multi_split         split clusters based on sample id, separator (default='C')
```

# Example
    python mcdevol.py -i <samfilesdir> -c <contigseq.fasta> -o <outputdir> --abundformat metabat -n 24

`-i` and `-a` options are mutually exclusive.

For multi-sample binning run

    python mcdevol.py -i <samfilesdir> -c <contigseq.fasta> -o <outputdir> --abundformat metabat -n 24 --multi-split


# Recommended
McDevol is most powerful when input contigs have high read coverage profiles.
