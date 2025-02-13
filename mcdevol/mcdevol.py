#!/usr/bin/env python

import os
import sys

# Adding parent path
parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
sys.path.insert(0, parent_path)

import gzip
import version
import argparse
import logging
import dataparsing as ps
import byol_model as byol_model
from clustering import cluster

BOLD = '\033[1m'
END = '\033[0m'
RED = '\033[91m'
GREEN = '\033[92m'

def check_inputs(parser, args):

	if not len(sys.argv) > 1:
		parser.print_help()
		sys.exit(0)

	if args.version:
		print("McDevol",version.__version__)
		sys.exit(0)

	""" abundance input parsing """
	if args.abundance is None and args.inputdir is None:
		print("Input missing! abundance file in tsv format -a/--abundance or --inputdir")
		sys.exit(1)
	if args.abundance is not None and args.inputdir is not None:
		print("Conflict arguments! Both abundance and input directory for sam files are given. Provide only one of them")
		sys.exit(1)
	if args.inputdir is not None:
		if not os.path.isdir(args.inputdir):
			print(f"Error: The directory {args.inputdir} does not exist.")
			sys.exit(1)
		else:
			sam_files = [f for f in os.listdir(args.inputdir) if f.endswith('.sam')]
			if  not sam_files:
				print(f"Error: The directory {args.inputdir} does not contains any samfiles.")
				sys.exit(1)
		if args.abundformat != 'std':
			print(f"WARNING: {args.abundformat} is incompatable with samfiles input. Mcdevol automatically creates abundance file in standard format")
	
	if args.abundformat not in ['std', 'metabat']:
		raise ValueError(f'Invalid abundance file format {args.abundformat}. Only std or metabat format is allowed')
	
	""" Assembled contig input validation """
	if args.contigs is None:
		print("Input missing! Please specifiy a contig assembly file with -c or --contigs")
		sys.exit(1)
	
	args.contigs = os.path.abspath(args.contigs)
	if args.contigs is gzip.GzipFile(args.contigs, 'r'):
		args.contigs = gzip.open(args.contigs,'rb')
	  
	if args.outdir is None:
		args.outdir = os.path.join(os.getcwd(), 'mcdevol_results/')
	else:
		args.outdir = os.path.join(args.outdir + '/')

	try:
		os.makedirs(args.outdir, exist_ok=True)
	except FileExistsError as e:
		print(f'Error: Could not create output directory {args.outdir}. {str(e)}')
		sys.exit(1)

	
	# BYOL parameters
	# TODO: As of now it is fixed to 6 for training
	if args.nfragments <=0 or not isinstance(args.nfragments, int):
		print(f"Invalid number of fragments: {args.nfragments}. It should be a positive integer")
		sys.exit(1)

	if args.readlength <=0 or not isinstance(args.readlength, int):
		print(f"Invalid read length: {args.readlength}. It should be a positive integer e.g., 250 for paired end")
		sys.exit(1)

def main():
	""" McDevol accurately reconstructs genomic bins from metagenomic samples using contig abundance and k-mer embedding """
	parser = argparse.ArgumentParser(prog='mcdevol', description="McDevol: An accurate metagenome binning of contigs based on decovolution of abundance and k-mer embedding\n")
	
	# input arguments
	# Create a mutually exclusive group for the -a and -i arguments
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("-a", "--abundance", type=str, help="abundance file in TSV format separated by tabs")
	group.add_argument("-i", "--inputdir", type=str, help="directory that contains SAM files")

	parser.add_argument("-c", "--contigs", type=str, help="contigs fasta (or zip)", required=True)
	parser.add_argument("-l", "--minlength", type=int, help="minimum length of contigs to be considered for binning", default=1000)
	parser.add_argument("-o", "--outdir", help="output directory")
	parser.add_argument("-n", "--ncores", help="Number of cores to use", default=os.cpu_count(), type=int)
	parser.add_argument("--abundformat", type=str, help="Format of abundance ('std|metabat', default='std')\
		std:[contigname, s1meancov, s2meancov,...,sNmeancov]; \
		metabat:[contigName, contigLen, totalAvgDepth, s1meancov, s1varcov,...,sNmeancov, sNvarcov]", default='std')
	parser.add_argument("-v", "--version", help="print version and exit", action="store_true")

	# fragmentation argument
	parser.add_argument("-f", "--nfragments", type=int, help="number of augumented fragments to generate", default=6)
	parser.add_argument("-r", "--readlength", type=int, help="average read length of fastq files", default=250)

	# model training argument
	parser.add_argument("-e", "--learningrate", type=float, help="learning rate", default=0.1)

	# clustering option
	parser.add_argument("--multi_split", help="split clusters based on sample id, separator (default='C')", action="store_true")

	args = parser.parse_args()

	check_inputs(parser, args)

	min_length: int = args.minlength
	abundance_format: str = args.abundformat
	nfragments: int = args.nfragments
	multi_split: bool = args.multi_split
	ncpus: int = args.ncores
	readlength: int = args.readlength
	learningrate: int = args.learningrate

	# logging
	logging.basicConfig(format='%(asctime)s - %(message)s', \
    level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S',
    filename= os.path.join(args.outdir, 'mcdevol.log'), filemode='w')
	logger = logging.getLogger()
	
	print(f'{BOLD}{GREEN}McDevol{END}{END} is running')
	logger.info(f'Metagenome binning started...')
	logger.info(f'contig file: {args.contigs}')
	logger.info(f'abundance file: {args.abundance}')
	logger.info(f'output directory: {args.outdir}')

	# load data
	outdir = args.outdir
	if args.abundance:
		abundance = args.abundance
	else:
		abundance = ps.generate_abundance(args.inputdir, ncpus, outdir)
	fasta_file = args.contigs

	numcontigs, contig_length, contig_names = ps.compute_kmerembeddings(outdir, fasta_file, min_length, logger, n_fragments=nfragments)
	logger.info(f'Kmer information is processed')
	abundance_matrix = ps.load_abundance(abundance, numcontigs, contig_names, min_length, logger, abundance_format)
	logger.info(f'Abundance information is processed')

	# byol training
	logger.info(f'Running BYOL entering')
	latent, contig_length, contig_names = byol_model.run(abundance_matrix, outdir, contig_length, contig_names, multi_split, ncpus, readlength, learningrate, nfragments) # type: ignore

	# leiden clustering
	logger.info(f'Running Leiden community detection')
	
	cluster(latent, contig_length, contig_names, fasta_file, outdir, ncpus, logger, multi_split)
	# learning_rate = [3e-2, 1e-1]
	# for lr in learning_rate:
	# 	logger.info(f'Running BYOL entering for {lr}')
	# 	latent, contig_length, contig_names = byol_model.run(abundance_matrix, outdir, contig_length, contig_names, multi_split, ncpus, readlength, lr) # type: ignore

	# 	# leiden clustering
	# 	logger.info(f'Running Leiden community detection for {lr}')
	# 	cluster_outdir = os.path.join(outdir,'lrate_'+str(lr)+'/')
	# 	try:
	# 		os.makedirs(cluster_outdir, exist_ok=True)
	# 	except FileExistsError as e:
	# 		print(f'Already {cluster_outdir} exist. {str(e)}')
	# 	cluster(latent, contig_length, contig_names, fasta_file, cluster_outdir, ncpus, logger, multi_split)

	# assembly
	logger.info(f'McDevol has generated metagenomic bins')
	print(f'{BOLD}{GREEN}McDevol binning{END}{END} is completed!')
if __name__ == "__main__":
	main()
