import os
import sys
import igraph as ig
import leidenalg
import pandas as pd
import numpy as np
import time
import hnswlib
import logging
import subprocess

util_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../util/'))
sys.path.insert(0, util_path)

def fit_hnsw_index(features, ncpus, ef: int = 100, M: int = 16,
                space: str = 'cosine', save_index_file: bool = False) -> hnswlib.Index:
    """
    Fit an HNSW index with the given features using the HNSWlib library; Convenience function to create HNSW graph.

    :param features: A list of lists containing the embeddings.
    :param ef: The ef parameter to tune the HNSW algorithm (default: 100).
    :param M: The M parameter to tune the HNSW algorithm (default: 16).
    :param space: The space in which the index operates (default: 'l2').
    :param save_index_file: The path to save the HNSW index file (optional).

    :return: The HNSW index created using the given features.

    This function fits an HNSW index to the provided features, allowing efficient similarity search in high-dimensional spaces.
    """

    num_elements = len(features)
    labels_index = np.arange(num_elements)
    EMBEDDING_SIZE = len(features[0])

    # Declaring index
    # possible space options are l2, cosine or ip
    p = hnswlib.Index(space=space, dim=EMBEDDING_SIZE)

    # Initing index - the maximum number of elements should be known
    p.init_index(max_elements=num_elements, ef_construction=ef, M=M)

    # Element insertion
    p.add_items(features, labels_index, num_threads=ncpus)

    # Controlling the recall by setting ef
    # ef should always be > k
    p.set_ef(ef)

    # If you want to save the graph to a file
    if save_index_file:
        p.save_index(save_index_file)

    return p

def run_leiden(latent_norm, ncpus, resolution_param = 1.0):

    num_elements = len(latent_norm)

    p = fit_hnsw_index(latent_norm, ncpus)

    if num_elements > 100:
        max_edges = 100
    else:
        max_edges = int(num_elements/2)
    
    ann_neighbor_indices, ann_distances = p.knn_query(latent_norm, max_edges + 1, num_threads=8)

    partgraph_ratio = 50
    bandwidth = np.median(ann_distances[:,1:]) * 0.1

    sources = np.repeat(np.arange(num_elements),max_edges)

    targets = ann_neighbor_indices[:,1:].flatten()
    weights = ann_distances[:,1:].flatten()
    dist_cutoff = np.percentile(weights, partgraph_ratio)
    selected_index = weights <= dist_cutoff
    sources = sources[selected_index]
    targets = targets[selected_index]
    weights = weights[selected_index]

    weights = np.exp(-weights / bandwidth)
    index = sources > targets
    sources = sources[index]
    targets = targets[index]
    weights = weights[index]
    edgelist = list(zip(sources, targets))
    g = ig.Graph(num_elements, edgelist)

    rbconf = leidenalg.RBConfigurationVertexPartition(g, weights=weights,resolution_parameter=resolution_param)
    optimiser = leidenalg.Optimiser()
    optimiser.optimise_partition(rbconf, n_iterations=-1)

    community_assignments = rbconf.membership

    return community_assignments

def cluster(
    latent: np.ndarray,
    contig_length: np.ndarray,
    contig_names: np.ndarray,
    fasta_file: str,
    outdir: str,
    ncpus: int,
    logger: logging.Logger,
    multi_split: bool = False,
    seperator: str = 'C'
    ) -> None:
    """
    Clusters contigs based on latent representations and optionally splits the clusters if sample-wise contigs are provided.

    Args:
        latent (np.ndarray): Latent representations of contigs.
        contig_length (np.ndarray): Array containing the lengths of the contigs.
        contig_names (np.ndarray): Array of contig names.
        fasta_file (str): Path to the input FASTA file containing contig sequences.
        outdir (str): Directory where the output files will be stored.
        ncpus (int): Number of CPUs to use for parallel processing.
        logger (logging.Logger): Logger object for logging progress and information.
        multi_split (bool, optional): Whether to split clusters sample-wise. Defaults to False.
        separator (str, optional): Separator used to split contig names for multi_split. Defaults to 'C'.

    Returns:
        None
    """
    start_time = time.time()
    latent_norm = latent / np.linalg.norm(latent, axis=1, keepdims=True)
    
    community_assignments = run_leiden(latent_norm, ncpus)

    cluster_ids = pd.DataFrame({
        "contig_name": contig_names, 
        "cluster_id": community_assignments
    })
    file_name = os.path.join(outdir, 'allbins.tsv')
    cluster_ids.to_csv(file_name, header=None, sep='\t', index=False)
    
    cluster_ids["contig_length"] = contig_length
    clustersize = cluster_ids.groupby("cluster_id")["contig_length"].sum().reset_index(drop=True)
    clusterids_selected = clustersize[clustersize>=200000].index
    cluster_selected = cluster_ids[cluster_ids['cluster_id'].isin(clusterids_selected)][["contig_name","cluster_id"]]
    logger.info(f'Filtered bins by 200kb size: {len(cluster_selected)}')
    file_name = 'bins_filtered.tsv'
    cluster_selected.to_csv(os.path.join(outdir, file_name), header=None, sep=',', index=False)

    if multi_split:
        clusters = cluster_selected.groupby("cluster_id")["contig_name"].apply(list).tolist()
        cluster_counter = 0
        sub_clusters = []
        for cluster in clusters:
            pd_data = pd.DataFrame(cluster, columns=["contig_name"])
            pd_data["sample_id"] = pd_data["contig_name"].str.split(seperator).str[0]
            split_data = pd_data.groupby("sample_id")["contig_name"].apply(list)
            for sub_cluster in split_data:
                sub_cluster = [(c, cluster_counter) for c in sub_cluster]
                sub_clusters.extend(sub_cluster)
                cluster_counter += 1
        file_name = 'cluster_split_allsamplewisebins'
        pd.DataFrame(sub_clusters).to_csv(os.path.join(outdir, file_name), header=None, sep=',', index=False)

        # save bins
        bindirectory = os.path.join(outdir,'cluster_split_bins/')
        # fetch sequences from contig fasta file
        subprocess.run(f"{util_path}/get_sequence_bybin {outdir} {file_name} {fasta_file} bin {bindirectory}", shell=True)

        pd_data = pd.DataFrame(contig_names, columns=["contig_name"])
        pd_data["sample_id"] = pd_data["contig_name"].str.split('C').str[0]
        pd_data["index"] = pd_data.index
        sampleindices = pd_data.groupby("sample_id")["index"].apply(list)
        bindirectory = os.path.join(outdir,'split_cluster_bins/')

        for i, inds in enumerate(sampleindices):
            latent_sample = latent_norm[inds]
            contig_length_sample = contig_length[inds]
            names_subset = contig_names[inds]
            community_assignments = run_leiden(latent_sample, 12)
            bin_ids = pd.DataFrame({
                "contig_name": names_subset,
                "cluster_id": community_assignments,
                "contig_length": contig_length_sample
            })
            binsize = pd.DataFrame(bin_ids.groupby("cluster_id")["contig_length"].sum().reset_index(drop=True))
            binids_selected = binsize[binsize>=200000].index
            bins_selected = bin_ids[bin_ids["cluster_id"].isin(binids_selected)][["contig_name","cluster_id"]]
            file_name = f'S{i}_bins_filtered'
            bins_selected.to_csv(os.path.join(outdir, file_name), header=None, sep=',', index=False)

            # fetch sequences from contig fasta file
            subprocess.run(f"{util_path}/get_sequence_bybin {outdir} {file_name} {fasta_file} bin {bindirectory}", shell=True)
        logger.info(f'Splitting clusters by sample: {len(cluster_selected)}')


    else:
        bindirectory = os.path.join(outdir+'bins/')
        # fetch sequences from contig fasta file
        subprocess.run(f"{util_path}/get_sequence_bybin {outdir} {file_name} {fasta_file} bin {bindirectory}", shell=True)

    logger.info(f'Clustering is completed in {time.time()-start_time:.2f} seconds')

# if __name__ == "__main__":
#     path = '/home/yazhini/work/cami2_datasets/marine/data_1kcontigs/MA_assembly/multi_split_assembly/byol_1/'
#     outpath = '/home/yazhini/work/cami2_datasets/marine/data_1kcontigs/MA_assembly/multi_split_assembly/byol_1/'
    
#     logging.basicConfig(format='%(asctime)s - %(message)s', \
#     level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S',
#     filename= outpath + '/clustering.log', filemode='w')
#     logger = logging.getLogger()
#     latent = np.load(path + 'latent.npy', allow_pickle=True) # 'byoltrlc_both_augsplit

#     contig_length = np.load(path + '../contigs_1klength.npz', allow_pickle=True)['arr_0']
#     contig_names = np.load(path + '../contigs_1knames.npz', allow_pickle=True)['arr_0']
#     cluster(latent, contig_length, contig_names, outpath, multi_split=True)
    # scmg_dict = pd.read_csv(path + '../marker_hits', header=None,sep='\t')

    # scmg_dict = dict(scmg_dict.groupby(1)[0].apply(list))

    # max_key = max(scmg_dict, key=lambda k: len(scmg_dict[k]))
    # scmg_freq ={k:len(scmg_dict[k]) for k in scmg_dict.keys()}
    # scmg_freq = sorted(scmg_freq.items(), key=lambda item: item[1])