__doc__ = """ preparing files required for binning """

import os
import sys
import tqdm
import time
import gc
import random
import logging
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Tuple, Dict
from Bio.SeqRecord import SeqRecord
from multiprocessing.pool import Pool
# Adding the `/util` directory to the system path
util_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../util/'))
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, util_path)
import kmer_counter


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

# Limit TensorFlow to a fraction of the total GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # To prevents TensorFlow from pre-allocating the entire GPU memory
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

def compute_kmerembeddings(
    outdir: str,
    fasta_file: str,
    min_length: int,
    logger: logging.Logger,
    n_fragments: int=6
    ) -> Tuple[int, np.ndarray, np.ndarray]:

    """
    Computes k-mer embeddings for sequences from a FASTA file, with optional fragmentation.

    Args:
        outdir (str): Directory where the output files will be stored.
        fasta_file (str): Path to the contig FASTA file.
        min_length (int): Minimum length of contigs to consider.
        logger (logging.Logger): Logger object for logging progress and information.
        n_fragments (int, optional): Number of fragments to generate per sequence. Defaults to 6.

    Returns:
        Tuple[int, np.ndarray, np.ndarray]: 
            - numcontigs (int): Number of contigs processed.
            - contig_length (np.ndarray): Array of lengths of the contigs.
            - contig_names (np.ndarray): Array of contig names.
    """

    # fragment sequences
    start_time = time.time()
    for i in range(n_fragments):
        output_file = outdir + 'fragments_'+ str(i) +'.fasta'

        fragments = []
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_length = len(record.seq)
            if seq_length >= min_length:
                start_pos = random.randint(0, seq_length - min_length)
                length = min_length + random.randint(0, seq_length - start_pos - min_length)
                end_pos = start_pos + length
                fragment = record.seq[start_pos:end_pos]
                fragment_id = f"{record.id} {start_pos}:{end_pos}"
                fragments.append(SeqRecord(Seq(fragment), id=fragment_id, description=""))
        
        SeqIO.write(fragments, output_file, "fasta")
    logger.info(f'Fragmentation is complete in {time.time()-start_time:.2f} seconds')

    """ information about kmer counter output
    information returned as tuple of 12 data (11 kmer counts and one contig names)
    (
        pre_l4n1mers.into_pyarray(py), -> length of contig
        pre_5mers.into_pyarray(py), -> 512D
        pre_4mers.into_pyarray(py), -> 136D
        pre_3mers.into_pyarray(py), -> 32D
        pre_2mers.into_pyarray(py), -> 10D
        pre_1mers.into_pyarray(py), -> 2D
        pre_10mers.into_pyarray(py), -> 528D
        pre_9mers.into_pyarray(py), -> 256D
        pre_8mers.into_pyarray(py), -> 136D
        pre_7mers.into_pyarray(py), -> 64D
        pre_6mers.into_pyarray(py), -> 36D
        contig_names
    ) """
    s = time.time()
    file_ids = [fasta_file] + [outdir+'fragments_'+str(f)+'.fasta' for f in range(n_fragments)]
    file_counter = [''] + list(range(1, n_fragments+1))

    # Create compositional model
    kmer_inputs = [Input(shape=(v,)) for v in [512,136,32,10,2,528,256,136,64,36]]
    x = layers.Concatenate()(kmer_inputs)
    x = layers.BatchNormalization()(x)

    for units in [1024 * 4, 1024 * 8 * 2]:
        x = layers.Dense(units, activation='tanh', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Dense(512, use_bias=False, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    kmermodel = Model([Input(shape=(136,)),*kmer_inputs], x)
    kmermodel.compile()
    # Load genomeface model weights
    path_weight = os.path.join(current_path, "genomeface_weights", "general_t2eval.m")
    kmermodel.load_weights(path_weight)
    
    for counter, infile in enumerate(file_ids):
        aaq = kmer_counter.find_nMer_distributions(infile, min_length)
        contig_names = np.asarray(aaq[-1])
        if counter == 0:
            contig_length = np.asarray(aaq[0])
            contig_length = contig_length[contig_length >= min_length]
        assert len(contig_length) == len(contig_names)
        numcontigs = len(contig_names)
        inpts = [np.reshape(aaq[i], (-1, size)).astype(np.float32) \
            for i, size in enumerate([512, 136, 32, 10, 2, 528, 256, 136, 64, 36], start=1)]

        # generate numcontigs x 136 array filled with zeros
        model_data_in = [np.zeros((inpts[0].shape[0], 136), dtype=np.float32)]
        for i in range(len(inpts)):
            model_data_in.append(inpts[i])
        with tf.device('/cpu:0'):
            datasets = [tf.data.Dataset.from_tensor_slices(arr) for arr in model_data_in]
        dataset = tf.data.Dataset.zip(tuple(datasets))
        batch_size = 4096
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        y20_cat = np.zeros((numcontigs, 512))
        filled = 0
        for _, b in enumerate(dataset):
            y20_cat[filled : filled + batch_size, :] = kmermodel.predict(x=b, verbose=0, batch_size=batch_size)
            filled += batch_size

        y20_cat /= np.linalg.norm(y20_cat, axis=1, keepdims=True)
        if counter == 0:
            np.save(os.path.join(outdir, f'kmer_embedding{file_counter[counter]}.npy'), y20_cat)
        else:
            np.save(os.path.join(outdir, f'kmer_embedding_augment{file_counter[counter]}.npy'), y20_cat)    
        tf.keras.backend.clear_session()
        gc.collect()
    
    logger.info(f'Embedding is complete in {time.time()-s:.2f} seconds')
    return numcontigs, contig_length, contig_names

def load_abundance(
    abundance_file: str,
    numcontigs: int,
    contig_names: np.ndarray,
    min_length: int,
    logger: logging.Logger,
    abundformat: str ='std'):
    """
    Loads and processes an abundance file into a numpy array.

    Args:
        abundance_file (str): Path to the abundance file in TSV format.
        numcontigs (int): Number of contigs expected in the abundance file.
        contig_names (List[str]): List of contig names in the expected order.
        min_length (int): Minimum length of contigs to include in the result.
        logger (logging.Logger): Logger object for logging progress and information.
        abundformat (str, optional): Format of the abundance file. Defaults to 'std'.
            - 'std': Standard format, assumes abundance data starts from the second column.
            - 'metabat': MetaBAT format, assumes additional columns for contig length, total coverage and variance.

    Returns:
        np.ndarray: A numpy array containing the abundance data ordered by `contig_names`.
    """
    try:
    # Attempt to read the file with tab separator
        pd.read_csv(abundance_file, sep='\t')
    except Exception as e:
        raise ValueError(f"Failed to parse the file as tab-separated: {e}")
    
    abundance_header = pd.read_csv(abundance_file, sep='\t', nrows=0)

    if len(abundance_header.columns) == 0:
        raise ValueError(f"abundance header is empty. Check your input abundance file!")
    
    names_dict: Dict[str, int] = {name: index for index, name in enumerate(contig_names)}
    s = time.time()
    if abundformat == 'std':
        num_samples =  len(abundance_header.columns) - 1

        arr = np.zeros((numcontigs, num_samples),dtype='float')
        logger.info(f'Loading abundance file with {numcontigs} contigs and {num_samples} samples')
        used = 0
        abundance_names = []
        reader = pd.read_csv(abundance_file, sep='\t',\
            lineterminator='\n', engine='c', chunksize=10**6)
        for chunk in tqdm.tqdm(reader):
            # TODO: this condition may not be needed as input need not have length column.
            # This should be handled well by ordered_indices selection
            # chunk_part = chunk[chunk['contigLen'] >= min_length]
            abundance_names.extend(chunk['contigName'].str.strip())
            arr[used:used+len(chunk)] = chunk.iloc[:,1:len(chunk.columns)].to_numpy()
            used += len(chunk)
    
        # Remove data for contigs shorter than min_length.
        # It would be present if abundance file is created from aligner2counts as it uses contigs length to save.
        # If not it should be processed here
        abundance_names = np.array(abundance_names)

        if len(abundance_names) > len(contig_names):
            indices = np.where(np.isin(abundance_names, contig_names))[0]
            arr = arr[indices]
            abundance_names = abundance_names[indices]

        # reorder abundance as per contigs order in sequence
        abundance_names_dict = {name: index for index, name in enumerate(abundance_names)}
        # ordered_indices = [names_dict[name] for name in abundance_names if name in names_dict]
        ordered_indices = [abundance_names_dict[name] for name in contig_names if name in abundance_names]

        arr = arr[ordered_indices]
        gc.collect()
        logger.info(f'Loaded abundance file in {time.time()-s:.2f} seconds')

        return arr

    if abundformat == 'metabat':
        num_columns = len(abundance_header.columns)
        num_samples =  int((num_columns - 3) // 2)
        logger.info(f'Loading abundance file with {numcontigs} contigs and {num_samples} samples')
        used = 0
        arr = [] # np.zeros((numcontigs,num_samples),dtype='float') 
        abundance_names = []

        reader = pd.read_csv(abundance_file, sep='\t', \
            lineterminator='\n', engine='c', chunksize=10**6)
        for chunk in tqdm.tqdm(reader):
            chunk_part = chunk[chunk['contigLen'] >= min_length]
            abundance_names.extend(chunk_part['contigName'].str.strip())
            arr.append(chunk_part.iloc[:,range(3,num_columns,2)].to_numpy())
            used += len(chunk_part)
        
        if arr:
            arr = np.vstack(arr)
        else:
            arr = np.array([])

        # reorder abundance as per contigs order in sequence
        contigs_names_filtered = [name for name in contig_names if name in abundance_names]

        abundance_names_dict = {name: index for index, name in enumerate(abundance_names)}
        ordered_indices = [abundance_names_dict[name] for name in contigs_names_filtered if name in abundance_names]
 
        arr = arr[ordered_indices]
        gc.collect()
        logger.info(f'Loaded abundance file in {time.time()-s:.2f} seconds')
        return arr

def process_sample(args: Tuple[str, str, str]) -> None:
    """
    Processes a single `.sam` file by running a shell command to generate counts.

    Args:
        args (Tuple[str, str, str]): A tuple containing:
            - sam (str): The name of the `.sam` file to be processed.
            - inputdir (str): The directory where the input `.sam` file is located.
            - outdir (str): The directory where the output files will be stored.

    Returns:
        None
    """
    sam, inputdir, outdir = args
    sample_id = sam.split('.')[0]
    subprocess.run(f"cat {os.path.join(inputdir, sam)} | {util_path}/aligner2counts {outdir} {sample_id}", shell=True)

def generate_abundance(inputdir: str, ncores: int, outdir: str) -> str:
    """
    Generates abundance data from multiple `.sam` files by processing them in parallel and 
    aggregating the coverage data.

    Args:
        inputdir (str): The directory containing the `.sam` files.
        ncores (int): The number of CPU cores to use for parallel processing.
        outdir (str): The directory where output files and the final abundance file will be stored.

    Returns:
        str: The path to the generated `abundance.tsv` file.
    """
    samfiles: List[str] = [f for f in os.listdir(inputdir) if f.endswith('.sam')]
    ncpu: int = min(ncores, len(samfiles))
    args: List[Tuple[str, str, str]] = [(sam, inputdir, outdir) for sam in samfiles]
    with Pool(ncpu) as pool:
        pool.map(process_sample, args)

    coverage_files: List[str] = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.endswith('_coverage')]
    coverage_data_list: List[pd.DataFrame] = []

    for coverage_file in coverage_files:
        df = pd.read_csv(coverage_file, header=None, sep=' ')
        df.columns = ['contigName', 'sampleID', 'coverage']
        coverage_data_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    coverage_data: pd.DataFrame = pd.concat(coverage_data_list)
    coverage_data = coverage_data.pivot_table(index = 'contigName', columns = 'sampleID', values = 'coverage').sort_index(axis=1)

    abundance_file: str = os.path.join(outdir, "abundance.tsv")
    coverage_data.to_csv(abundance_file, sep='\t', index=False)

    return abundance_file

if __name__ == "__main__":
    pass