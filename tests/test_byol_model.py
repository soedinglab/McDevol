
import os
import sys

parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
sys.path.insert(0, parent_path)
sys.path.insert(0, os.path.join(parent_path, 'mcdevol'))

import unittest
import numpy as np
import torch
import logging
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from byol_model import BYOLmodel

class TestBYOLModelTraining(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

        self.abundance_matrix = np.random.rand(100, 20).astype(np.float32)  # 100 contigs, 20 samples
        self.kmer_data = {
            'kmer': np.random.rand(100, 128).astype(np.float32),
            'kmeraug1': np.random.rand(100, 128).astype(np.float32),
            'kmeraug2': np.random.rand(100, 128).astype(np.float32),
            'kmeraug3': np.random.rand(100, 128).astype(np.float32),
            'kmeraug4': np.random.rand(100, 128).astype(np.float32),
            'kmeraug5': np.random.rand(100, 128).astype(np.float32),
            'kmeraug6': np.random.rand(100, 128).astype(np.float32),
        }
        self.contig_length = np.random.rand(100).astype(np.float32)
        self.outdir = "/tmp"
        self.logger = logging.getLogger("test_logger")
        self.multi_split = True
        self.ncpus = 4
        self.model_path = os.path.join(self.outdir, 'byol_model.pth')

        # Instantiate the model
        self.model = BYOLmodel(
            abundance_matrix=self.abundance_matrix,
            kmer_data=self.kmer_data,
            contig_length=self.contig_length,
            outdir=self.outdir,
            logger=self.logger,
            multi_split=self.multi_split,
            ncpus=self.ncpus,
            seed=42
        )

    def test_data_augment(self):
        """ Test data augmentation logic """
        rcounts = torch.rand(100, 20).float()
        contigs_length = torch.rand(100).float() * 1000
        fraction_pi = 0.5
        augmented_data = self.model.data_augment(rcounts, contigs_length, fraction_pi)

        self.assertEqual(augmented_data.shape, rcounts.shape)
        self.assertTrue(torch.all(augmented_data >= 0))

        rawread_counts = torch.rand(100, 20).float()
        contigs_length = torch.rand(100).float() * 100000
        fraction_pi = 0.6
        kmers = [torch.rand(100, 128).float() for _ in range(6)]
        normalized_kmers = [F.normalize(kmer, p=2, dim=-1) for kmer in kmers]
        augmented_online, augmented_target, kmeraug_online, kmeraug_target = self.model.apply_augmentations(
            rawread_counts, contigs_length, fraction_pi, normalized_kmers
        )
        self.assertEqual(augmented_online.shape, rawread_counts.shape)
        self.assertEqual(augmented_target.shape, rawread_counts.shape)
        self.assertEqual(kmeraug_online.shape, kmeraug_target.shape)

    def test_trainmodel(self):
        nepochs = 2
        batchsteps = [1]

        self.model.trainmodel(nepochs=nepochs, batchsteps=batchsteps)

        for param in self.model.parameters():
            self.assertTrue(param.grad is None or not torch.all(param.grad == 0))

        self.assertTrue(torch.load(self.model_path))

    def test_initialize_target_network(self):
        self.model.initialize_target_network()

        # Check that target network parameters are initialized and frozen
        self.assertIsNotNone(self.model.target_encoder)
        self.assertIsNotNone(self.model.target_projector)

        for param in self.model.target_encoder.parameters():
            self.assertFalse(param.requires_grad)
        for param in self.model.target_projector.parameters():
            self.assertFalse(param.requires_grad)

    def test_update_moving_average(self):   
        
        self.model.initialize_target_network()

        initial_encoder_params = [param.clone() for param in self.model.target_encoder.parameters()]
        initial_projector_params = [param.clone() for param in self.model.target_projector.parameters()]
        nepochs = 4
        batchsteps = [1]

        self.model.trainmodel(nepochs=nepochs, batchsteps=batchsteps)
        self.model.update_moving_average()

        # Ensure parameters have changed (moving average update)
        for param, initial_param in zip(self.model.target_encoder.parameters(), initial_encoder_params):
            self.assertFalse(torch.equal(param, initial_param))
        for param, initial_param in zip(self.model.target_projector.parameters(), initial_projector_params):
            self.assertFalse(torch.equal(param, initial_param))


    def test_compute_loss(self):
        """ Test the computation of the BYOL loss """
        z1 = torch.rand(32, 128).float()
        z2 = torch.rand(32, 128).float()

        loss = self.model.compute_loss(z1, z2)

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar
        self.assertGreaterEqual(loss.item(), 0) 

    def test_process_batches(self):
        epoch = 10
        batchsteps = [50, 100, 150]
        dataloader = DataLoader(self.model.dataset_train, batch_size=16)
        training = True

        loss_array = []
        latent_space = []

        self.model.initialize_target_network()
        self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.scheduler = CosineAnnealingLR(self.model.optimizer, T_max=10-5, eta_min=0)

        
        # Process one batch (using mock logger to prevent actual log output)
        self.model.process_batches(epoch, dataloader, training, loss_array, latent_space, 0.7)

        # Ensure loss is being calculated and stored
        self.assertGreater(len(loss_array), 0)
        self.assertGreater(len(latent_space), 0)

        if training and epoch % 10 == 0:  # Save every 10 epochs, for example
            torch.save(self.model.state_dict(), self.model_path)

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == "__main__":
    unittest.main()

