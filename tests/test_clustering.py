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
import tempfile
import logging
from unittest.mock import patch, MagicMock, call
import clustering
from clustering import cluster, run_leiden

class TestClusterFunction(unittest.TestCase):
    def setUp(self):
        # 100 contigs, 32-dimensional latent space
        self.latent = np.random.rand(100, 32)
        self.contig_length = np.random.randint(1000, 10000, 100)
        self.contig_names = np.array([f"contig_{i}" for i in range(100)])
        self.fasta_file = "dummy.fasta"
        self.outdir = tempfile.mkdtemp()
        self.ncpus = 2
        self.logger = logging.getLogger("test_logger")

    @patch('clustering.run_leiden')
    @patch('subprocess.run')
    # @patch('sys.stdout', new_callable=io.StringIO)
    def test_cluster(self, mock_subprocess_run, mock_run_leiden):
        # Mock the Leiden clustering result
        # mock_run_leiden.return_value = np.random.randint(0, 20, 100)
        num_elements = 100
        mock_edgelist = [(i, i + 1) for i in range(num_elements - 1)]  # Simple chain graph
        mock_g = ig.Graph(num_elements, mock_edgelist)
        mock_run_leiden.return_value = (
            np.random.randint(0, 20, 100),  # Mocked community_assignments
            100,                            # Mocked num_elements
            50,                             # Mocked max_edges
            np.random.rand(100, 50),        # Mocked ann_distances
            np.random.randint(0, 100, (100, 50)),  # Mocked ann_neighbor_indices
            mock_g
        )
         
        cluster(self.latent, self.contig_length, self.contig_names, 
                self.fasta_file, self.outdir, self.ncpus, self.logger)

        # Check if files are created
        self.assertTrue(os.path.exists(os.path.join(self.outdir, 'allbins.tsv')))
        self.assertTrue(os.path.exists(os.path.join(self.outdir, 'bins_filtered.tsv')))

        # Check if subprocess was called
        mock_subprocess_run.assert_called_once()

    def tearDown(self):
        # Clean up temporary directory
        for file in os.listdir(self.outdir):
            os.remove(os.path.join(self.outdir, file))
        os.rmdir(self.outdir)

class TestClusterFunction(unittest.TestCase):
    @patch('subprocess.run')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.print')
    @patch('pandas.DataFrame.to_csv')
    @patch('clustering.run_leiden')  # Replace 'clustering' with the actual module name
    def test_cluster_with_multi_split(
        self, mock_run_leiden, mock_to_csv, mock_print, mock_exists, mock_makedirs, mock_subprocess_run
    ):
        # Mock inputs
        latent = np.random.rand(100, 10)
        contig_length = np.random.randint(1000, 5000, size=100)
        contig_names = np.array([f"S1C{i}" for i in range(50)] + [f"S2C{i}" for i in range(50, 100)])
        fasta_file = 'test.fasta'
        outdir = 'test_output'
        ncpus = 4
        logger = logging.getLogger('test_logger')
        multi_split = True
        separator = 'C'

        mock_exists.return_value = False
        mock_to_csv.return_value = None

        def dynamic_run_leiden(latent_sample, *args, **kwargs):
            num_elements = len(latent_sample)
            return (
                np.random.randint(0, 10, size=num_elements),
                num_elements,
                100,
                [np.random.rand(5) for _ in range(num_elements)],
                [np.random.randint(0, num_elements, size=5) for _ in range(num_elements)],
                MagicMock(vcount=lambda: num_elements)
            )

        mock_run_leiden.side_effect = dynamic_run_leiden

        # Call the function
        from clustering import cluster  # Replace 'clustering' with your actual module name
        cluster(latent, contig_length, contig_names, fasta_file, outdir, ncpus, logger, multi_split, separator=separator)

        # Verify that run_leiden was called with appropriate subsets
        calls = mock_run_leiden.call_args_list
        self.assertGreater(len(calls), 0, "Expected multiple calls to run_leiden for sample-wise clustering.")
        

        # Check calls for critical operations
        self.assertTrue(mock_makedirs.called)
        self.assertTrue(mock_subprocess_run.called)
        self.assertTrue(mock_to_csv.called)

        # Verify that cluster splitting was performed
        split_calls = [call for call in mock_to_csv.call_args_list if 'cluster_split_allsamplewisebins' in str(call)]
        self.assertGreater(len(split_calls), 0, "Expected 'cluster_split_allsamplewisebins' to be saved.")

        # Verify sub-clustering logic
        bin_calls = [call for call in mock_subprocess_run.call_args_list if "get_sequence_bybin" in str(call)]
        self.assertGreater(len(bin_calls), 0, "Expected 'get_sequence_bybin' to be called for sample bins.")


if __name__ == '__main__':
    unittest.main()