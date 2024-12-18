import os
import sys

parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
sys.path.insert(0, parent_path)
sys.path.insert(0, os.path.join(parent_path, 'mcdevol'))

import unittest
import numpy as np
import pandas as pd # type: ignore
import logging
import tempfile
import os
from io import StringIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Assuming the function is in a module named 'abundance_loader'
from dataparsing import load_abundance, compute_kmerembeddings



class TestComputeKmerEmbeddings(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock FASTA file
        self.fasta_file = os.path.join(self.test_dir, "test.fasta")
        self.create_mock_fasta()
        parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weight_path = os.path.join(parent_path, "mcdevol", "genomeface_weights", "general_t2eval.m.index")
        self.assertTrue(os.path.exists(weight_path), f"Weight file not found at {weight_path}")

        # Set up logger
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)

    def create_mock_fasta(self):
        # Create mock DNA sequences
        sequences = [
            ("contig1", "ATCGATCGATCGATCGATCG"), # 20 bp
            ("contig2", "GCTAGCTAGCTAGCTAGCTAGCTAGA"), # 26 bp
            ("contig3", "TATATATATATATATA") # 16 bp
        ]
        
        # Write sequences to FASTA file
        with open(self.fasta_file, "w") as handle:
            for seq_id, seq in sequences:
                record = SeqRecord(Seq(seq), id=seq_id, description="")
                SeqIO.write(record, handle, "fasta")

    def test_compute_kmerembeddings(self):
        min_length = 20
        n_fragments = 2
        
        numcontigs, contig_length, contig_names = compute_kmerembeddings(
            self.test_dir,
            self.fasta_file,
            min_length,
            self.logger,
            n_fragments
        )
        
        # Check the number of contigs
        self.assertEqual(numcontigs, 2)  # Only 2 contigs should meet the min_length requirement
        
        # Check contig lengths
        np.testing.assert_array_equal(contig_length, np.array([20, 26]))
        
        # Check contig names
        np.testing.assert_array_equal(contig_names, np.array(["contig1", "contig2"]))
        
        # Check if embedding files are created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "kmer_embedding.npy")))
        self.assertTrue(os.path.join(self.test_dir, "kmer_embedding_augment1.npy"))
        self.assertTrue(os.path.join(self.test_dir, "kmer_embedding_augment2.npy"))

    def tearDown(self):
        # Remove the temporary directory and its contents
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)


class TestLoadAbundance(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        
        # Sample data for testing
        self.sample_data = """contigName\tsample1\tsample2\tsample3
        contig1\t1.0\t2.0\t3.0
        contig2\t4.0\t5.0\t6.0
        contig3\t7.0\t8.0\t9.0
        """
        
        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.temp_file.write(self.sample_data)
        self.temp_file.flush()
        self.temp_file.close()
        
        self.contig_names = np.array(['contig1', 'contig2', 'contig3'])

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_load_abundance_standard_format(self):

        # with open(self.temp_file.name, 'r') as f:
        #     print(f"File content:\n{f.read()}")

        result = load_abundance(
            self.temp_file.name,
            numcontigs=3,
            contig_names=self.contig_names,
            min_length=0,
            logger=self.logger
        )
        
        expected = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]])
        
        np.testing.assert_array_equal(result, expected)

    def test_load_abundance_file_not_found(self):
        with self.assertRaises(ValueError):
            load_abundance(
                'non_existent_file.tsv',
                numcontigs=3,
                contig_names=self.contig_names,
                min_length=0,
                logger=self.logger
            )

    def test_load_abundance_empty_file(self):
        empty_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        empty_file.close()
        
        with self.assertRaises(ValueError):
            load_abundance(
                empty_file.name,
                numcontigs=3,
                contig_names=self.contig_names,
                min_length=0,
                logger=self.logger
            )
        
        os.unlink(empty_file.name)

    def test_load_abundance_mismatched_contigs(self):
        mismatched_contig_names = np.array(['contig2', 'contig1', 'contig4'])
        
        result = load_abundance(
            self.temp_file.name,
            numcontigs=3,
            contig_names=mismatched_contig_names,
            min_length=0,
            logger=self.logger
        )

        # Expect only data for contig1 and contig2 to be present
        expected = np.array([[4.0, 5.0, 6.0],
                             [1.0, 2.0, 3.0]])
        
        np.testing.assert_array_equal(result, expected)


class TestLoadAbundanceMetaBAT(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        
        # Sample data for testing MetaBAT format
        self.sample_data = \
        """contigName\tcontigLen\ttotalAvgDepth\tcov1\tvar1\tcov2\tvar2\tcov3\tvar3
        contig1\t1000\t6.0\t1.0\t0.1\t2.0\t0.2\t3.0\t0.3
        contig2\t2000\t15.0\t4.0\t0.2\t5.0\t0.3\t6.0\t0.4
        contig3\t500\t24.0\t7.0\t0.3\t8.0\t0.4\t9.0\t0.5
        contig4\t1500\t33.0\t10.0\t0.4\t11.0\t0.5\t12.0\t0.6
        """

        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.temp_file.write(self.sample_data)
        self.temp_file.flush()
        self.temp_file.close()
        
        self.contig_names = np.array(['contig1', 'contig2', 'contig3', 'contig4'])

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_load_abundance_metabat_format(self):
        result = load_abundance(
            self.temp_file.name,
            numcontigs=4,
            contig_names=self.contig_names,
            min_length=1000,  # This should exclude contig3
            logger=self.logger,
            abundformat='metabat'
        )
        
        # Expected result: only contigs >= 1000 bp, and only depth columns
        expected = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [10.0, 11.0, 12.0]
        ])
        
        np.testing.assert_array_almost_equal(result, expected)
        self.assertEqual(result.shape, (3, 3))  # 3 contigs (excluding contig3), 3 samples

    def test_load_abundance_metabat_all_contigs(self):
        result = load_abundance(
            self.temp_file.name,
            numcontigs=4,
            contig_names=self.contig_names,
            min_length=0,  # This should include all contigs
            logger=self.logger,
            abundformat='metabat'
        )
        
        # Expected result: all contigs, only depth columns
        expected = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        
        np.testing.assert_array_almost_equal(result, expected)
        self.assertEqual(result.shape, (4, 3))  # 4 contigs, 3 samples

    def test_load_abundance_metabat_reordering(self):
        # Test with a different order of contig_names
        reordered_contig_names = np.array(['contig2', 'contig4', 'contig1', 'contig3'])
        
        result = load_abundance(
            self.temp_file.name,
            numcontigs=4,
            contig_names=reordered_contig_names,
            min_length=0,
            logger=self.logger,
            abundformat='metabat'
        )
        
        # Expected result: reordered according to reordered_contig_names
        expected = np.array([
            [4.0, 5.0, 6.0],
            [10.0, 11.0, 12.0],
            [1.0, 2.0, 3.0],
            [7.0, 8.0, 9.0]
        ])
        
        np.testing.assert_array_almost_equal(result, expected)
        self.assertEqual(result.shape, (4, 3))  # 3 contigs, 3 samples



if __name__ == '__main__':
    unittest.main()
