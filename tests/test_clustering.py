import os
import sys

parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
sys.path.insert(0, parent_path)
sys.path.insert(0, os.path.join(parent_path, 'mcdevol'))


import unittest
import numpy as np
import pandas as pd # type: ignore
import os
import io
import sys
import tempfile
import logging
import shutil
import igraph as ig
from io import StringIO
from unittest.mock import patch, MagicMock, call
import clustering
from clustering import cluster, run_leiden

class TestClusterFunction(unittest.TestCase):
    def setUp(self):
        # 100 contigs, 32-dimensional latent space
        self.latent = np.random.rand(100, 32)
        self.contig_length = np.random.randint(1000, 10000, 100)
        self.contig_names = np.array([f"k141_{i}" for i in range(100)])
        self.fasta_file = "dummy.fasta"
        self.outdir = tempfile.mkdtemp()
        self.ncpus = 2
        self.logger = logging.getLogger("test_logger")

    @patch('clustering.run_leiden')
    @patch('subprocess.run')
    def test_cluster(self, mock_subprocess_run, mock_run_leiden):
        num_elements = 100
        mock_edgelist = [(i, i + 1) for i in range(num_elements - 1)]  # Simple chain graph
        mock_run_leiden.return_value = (
            np.random.randint(0, 20, 100)
        )
         
        cluster(self.latent, self.contig_length, self.contig_names, 
                self.fasta_file, self.outdir, self.ncpus, self.logger)

        # Check if files are created
        self.assertTrue(os.path.exists(os.path.join(self.outdir, 'allbins.tsv')))
        self.assertTrue(os.path.exists(os.path.join(self.outdir, 'bins_filtered.tsv')))

        mock_subprocess_run.assert_called_once()

    def tearDown(self):
        for file in os.listdir(self.outdir):
            os.remove(os.path.join(self.outdir, file))
        os.rmdir(self.outdir)

def dynamic_run_leiden(latent_subset, ncpus, resolution_param=1.0, max_edges=100):
    num_elements = latent_subset.shape[0]
    community_assignments = np.random.randint(0, 20, size=num_elements)  # Simulate cluster IDs
    return community_assignments

class TestClusterFunctionMultiSplit(unittest.TestCase):

    def setUp(self):
        self.latent = np.random.rand(100, 32)
        self.contig_length = np.random.randint(1000, 10000, 100)
        self.contig_names = np.array([f"S1Ck141_{i}" for i in range(50)] + [f"S2C{i}" for i in range(50, 100)])
        self.fasta_file = "dummy.fasta"
        self.outdir = tempfile.mkdtemp()
        self.ncpus = 2
        self.logger = logging.getLogger("test_logger")

    @patch('clustering.run_leiden')
    @patch('subprocess.run')
    def test_cluster(self, mock_subprocess_run, mock_run_leiden):
        mock_run_leiden.side_effect = dynamic_run_leiden
        
        cluster(self.latent, self.contig_length, self.contig_names,
            self.fasta_file, self.outdir, self.ncpus, self.logger, True)

        self.assertEqual(mock_run_leiden.call_count, 3, "Expected run_leiden to be called once for the entire dataset and twice for two samples.")
        
        # Verify run_leiden was called for each subset
        calls = mock_run_leiden.call_args_list
        for i, call in enumerate(mock_run_leiden.call_args_list):
            latent_subset = call[0][0]  # Get the `latent_norm` argument from the call
            sample_size = latent_subset.shape[0]
            print(f"Call {i}: sample size = {sample_size}")
            self.assertTrue(sample_size in [100,50], "Each latent_subset should have size 50 (one for each sample).")

        self.assertTrue(os.path.exists(os.path.join(self.outdir, 'cluster_split_allsamplewisebins')))
            
        def tearDown(self):
            for file in os.listdir(self.outdir):
                file_path = os.path.join(self.outdir, file)
                if os.path.isfile(file_path):  # Delete files
                    os.remove(file_path)
                elif os.path.isdir(file_path):  # Delete directories
                    shutil.rmtree(file_path)
            os.rmdir(self.outdir)

if __name__ == '__main__':
    unittest.main()