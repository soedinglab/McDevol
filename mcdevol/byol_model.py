#!/usr/bin/env python
""" run byol training """

import os
import time
import copy
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from numpy import random as np_random
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
import logging
from torch.cuda.amp import autocast, GradScaler
# from lightning import Fabric

# TODO: speedup the process
# autocast and GradScaler helps # worked
# deepseed using fabric didn't speed up with single device
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(1.0, (self.last_epoch + 1) / self.total_iters) for base_lr in self.base_lrs]


# Combine both schedulers
class WarmUpThenScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_scheduler, main_scheduler):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_finished = False
        super(WarmUpThenScheduler, self).__init__(optimizer)
    
    def get_lr(self):
        if not self.warmup_finished:
            if self.warmup_scheduler.last_epoch < self.warmup_scheduler.total_iters:
                lr = self.warmup_scheduler.get_lr()
            else:
                self.warmup_finished = True
                lr = self.main_scheduler.get_lr()
                self.main_scheduler.last_epoch = self.warmup_scheduler.last_epoch - self.warmup_scheduler.total_iters
        else:
            lr = self.main_scheduler.get_lr()
        return lr
    
    def step(self, epoch=None):
        if not self.warmup_finished:
            self.warmup_scheduler.step(epoch)
        else:
            self.main_scheduler.step(epoch)

def normalize_counts(counts: np.ndarray):
    """ normalize count by mean division """
    counts_norm = counts / counts.mean(axis=1, keepdims=True)

    return counts_norm


def drawsample_frombinomial(counts, fraction_pi):
    """ augment data using binomial distribution """
    # floor_counts = np_random.binomial(\
    #     np.floor(counts.detach().cpu().numpy()).astype(int), fraction_pi)
    ceil_counts = np_random.binomial(\
        np.ceil(counts.detach().cpu().numpy()).astype(int), fraction_pi)
    # sample_counts = torch.from_numpy(\
        # ((floor_counts + ceil_counts) / 2).astype(np.float32)).to(counts.device)
    sample_counts = torch.from_numpy(ceil_counts.astype(np.float32)).to(counts.device)
    # if torch.min(sample_counts.sum(axis=1)) == 0.0:
    #     raise ValueError("sample_counts is zero counts across all samples")

    return sample_counts

def split_dataset(dataset, flag_test):
    """ split dataset into training and validation. Also, test set (if flag_test is true)"""
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    if flag_test:
        val_size = np.ceil((total_size - train_size) / 2).astype(int)
        test_size = np.floor((total_size - train_size) / 2).astype(int)
        # return random_split(dataset, [train_size, val_size, test_size])
        return random_split(dataset, [train_size, total_size-train_size, 0])
    else:
        val_size = np.ceil(total_size - train_size).astype(int)
        return random_split(dataset, [train_size, val_size])


def MLP(dim, projection_size, hidden_size=4096):
    " return multiple linear perceptron layer "

    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.LeakyReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        """ update target network by moving average of online network """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class EarlyStopper:
    """ early stop the model when validation loss increases """
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """ check if validation loss increases """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def target_update_moving_average(ema_updater, online_encode, \
    online_project, target_encode, target_project):
    for online_params1, online_params2, target_params1, target_params2 \
        in zip(online_encode.parameters(), online_project.parameters(), \
        target_encode.parameters(), target_project.parameters()):
        target_encode.data = ema_updater.update_average(target_params1.data, online_params1.data)
        target_project.data = ema_updater.update_average(target_params2.data, online_params2.data)

class BYOLmodel(nn.Module):
    """ train BYOL model """
    def __init__(
        self,
        abundance_matrix: np.ndarray,
        kmer_data: dict,
        contig_length:  np.ndarray,
        outdir: str,
        logger: object,
        multi_split: bool,
        ncpus: int = 8,
        seed: int = 0,
        lrate: float = 3e-6,
    ):

        super(BYOLmodel, self).__init__()
        torch.manual_seed(seed)

        if not abundance_matrix.dtype == np.float32:
            abundance_matrix = abundance_matrix.astype(np.float32)

        # Initialize simple attributes
        self.ncontigs, self.nsamples = abundance_matrix.shape
        self.multi_split = multi_split
        self.outdir = outdir
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = ncpus if torch.cuda.is_available() else 1
        self.usecuda = self.device == 'cuda'
        self.scheduler = None
        self.scaler = None
        self.augmentsteps = [0.9, 0.7, 0.6, 0.5] #, 0.3]

        # Model architecture parameters
        self.nhidden = 1024
        self.nlatent = 512 # 256 #
        self.dropout = 0.1
        projection_size = 256
        projection_hidden_size = 4096

        self.read_counts = torch.from_numpy(normalize_counts(abundance_matrix))
        self.rawread_counts = torch.from_numpy(abundance_matrix)
        self.contigs_length = torch.from_numpy(contig_length)

        for key, value in kmer_data.items():
            setattr(self, key, torch.from_numpy(value))

        self.indim = self.nsamples + self.kmeraug1.shape[1]
        self.dataset = TensorDataset(self.read_counts, self.rawread_counts,\
            self.contigs_length, self.kmer, self.kmeraug1, self.kmeraug2,\
            self.kmeraug3, self.kmeraug4, self.kmeraug5, self.kmeraug6)
        self.dataset_train, self.dataset_val, \
            self.dataset_test = split_dataset(self.dataset, True)

        # Define encoder and projector
        self.online_encoder = nn.Sequential(
            nn.Linear(self.indim, self.nhidden),
            nn.BatchNorm1d(self.nhidden),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.nhidden, self.nhidden),
            nn.BatchNorm1d(self.nhidden),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.nhidden, self.nlatent) # latent layer
        )
        self.online_projector = MLP(self.nlatent, projection_size, projection_hidden_size)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        
        # Define Target network setup
        self.use_momentum = True
        self.target_encoder = None
        self.target_projector = None
        moving_average_decay = 0.99
        self.target_ema_updater = EMA(moving_average_decay)

        if self.usecuda:
            self.cuda()
            print('Using device:', self.device)

            #Additional Info when using cuda
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

        self.optimizer = Adam(self.parameters(), lr=lrate, weight_decay=1e-6)
        self.scaler = GradScaler()

    def data_augment(
        self, 
        rcounts: torch.Tensor,
        contigs_length: torch.Tensor,
        fraction_pi: float):
        """ augment read counts """
        rcounts_sampled = rcounts.clone().detach()

        # if not self.multi_split:
        #     condition_short = (contigs_length > 4000) & (contigs_length <= 8000) # used for gs pooled assembly
        #     # condition_short = (contigs_length > 3000) & (contigs_length <= 8000) # used for pooled assembly
        #     condition_medium1 = (contigs_length > 8000) & (contigs_length <= 16000)
        #     condition_medium2 = (contigs_length > 16000) & (contigs_length <= 30000)
        #     condition_long = contigs_length > 30000
        
        # # for multisplit binning
        # else:
        #     condition_short = (contigs_length > 50000) & (contigs_length <= 100000)
        #     condition_medium1 = (contigs_length > 100000) & (contigs_length <= 500000)
        #     condition_medium2 = (contigs_length > 500000) & (contigs_length <= 800000)
        #     condition_long = contigs_length > 800000

        # if condition_short.any():
        #     # always samples short contigs with 0.9 fraction
        #     rcounts[condition_short] = drawsample_frombinomial(rcounts[condition_short], 0.9)
        # if condition_medium1.any():
        #     # always samples medium contigs with 0.7 fraction
        #     rcounts[condition_medium1] = drawsample_frombinomial(rcounts[condition_medium1], 0.7)
        # if condition_medium2.any():
        #     # always samples medium contigs with 0.6 fraction
        #     rcounts[condition_medium2] = drawsample_frombinomial(rcounts[condition_medium2], 0.6)
        # if condition_long.any():
        #     # sample longer contigs based on fraction_pi passed to the function
        #     rcounts[condition_long] = drawsample_frombinomial(rcounts[condition_long], fraction_pi)

        # Define contig length conditions
        if not self.multi_split:
            length_conditions = {
                'short': (contigs_length > 3000) & (contigs_length <= 8000),
                'medium1': (contigs_length > 8000) & (contigs_length <= 16000),
                'medium2': (contigs_length > 16000) & (contigs_length <= 30000),
                'long': (contigs_length > 30000)
            }
        else:
            length_conditions = {
                'short': (contigs_length > 50000) & (contigs_length <= 100000),
                'medium1': (contigs_length > 100000) & (contigs_length <= 500000),
                'medium2': (contigs_length > 500000) & (contigs_length <= 800000),
                'long': (contigs_length > 800000)
            }

        # Define fractions for each condition
        sampling_fractions = {
            'short': 0.9,
            'medium1': 0.7,
            'medium2': 0.6,
            'long': fraction_pi
        }

        # # Adjust sampling fractions based on the value of fraction_pi
        # if fraction_pi == 0.9:
        #     sampling_fractions = {key: 0.9 for key in sampling_fractions}  # All fractions set to 0.9
        # elif fraction_pi == 0.7:
        #     sampling_fractions.update({'medium1': 0.7, 'medium2': 0.7, 'long': 0.7})  # Set all except 'short' to 0.7
        # elif fraction_pi == 0.6:
        #     sampling_fractions.update({'medium2': 0.6, 'long': 0.6})  # Set 'medium2' and 'long' to 0.6
        # elif fraction_pi == 0.5:
        #     sampling_fractions['long'] = 0.5  # Only 'long' is set to 0.5

        # Apply sampling augmentation based on conditions
        for key, condition in length_conditions.items():
            if condition.any():
                rcounts_sampled[condition] = drawsample_frombinomial(rcounts_sampled[condition], sampling_fractions[key])

        zeroindices = (rcounts_sampled.sum(axis=1)==0.0).nonzero()
        
        rcounts_sampled[zeroindices] = rcounts[zeroindices].clone().detach()
        if torch.min(rcounts_sampled.sum(axis=1)) == 0.0:
            raise ValueError("sample_counts is zero counts across all samples")
        return normalize_counts(rcounts_sampled).to(contigs_length[0].device)

    def initialize_target_network(self):
        """ initialize target network """
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p, q in zip(self.target_encoder.parameters(), self.target_projector.parameters()):
            p.requires_grad = False
            q.requires_grad = False

    def update_moving_average(self):
        """ update target network by moving average """

        target_update_moving_average(self.target_ema_updater, self.online_encoder, \
            self.online_projector, self.target_encoder, self.target_projector)

    def forward(self, x, xt, pair=False):
        """ forward BYOL """
        latent = self.online_encoder(x)

        z1_online = self.online_predictor(self.online_projector(latent))
        z2_online = self.online_predictor(self.online_projector(self.online_encoder(xt)))

        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(xt)) # type: ignore
            z2_target =  self.target_projector(self.target_encoder(x)) # type: ignore

        if pair:
            byol_loss = self.compute_pairloss(z1_online, z1_target.detach()) + \
            self.compute_pairloss(z2_online, z2_target.detach()) # to symmetrize the loss
        else:
            byol_loss = self.compute_loss(z1_online, z1_target.detach()) + \
                self.compute_loss(z2_online, z2_target.detach()) # to symmetrize the loss

        return latent, byol_loss

    def compute_loss(self, z1, z2):
        """ loss for BYOL """
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        return torch.mean(2 - 2 * (z1 * z2.detach()).sum(dim=-1))
    
    def compute_pairloss(self, z1, z2):
        """ loss for BYOL """
        eta = 1e-1
        distance = torch.norm(z1 - z2, dim=-1, p=2)
        distance = torch.clamp(distance, min=1e-12)
        distance_inv = torch.reciprocal(distance)
        loss_term = torch.reciprocal(1 + distance_inv)
        return - eta * torch.mean(loss_term)

    def process_batches(self,
        epoch: int,
        dataloader,
        training: bool,
        *args):
        """ process batches """

        epoch_losses = []
        epoch_loss = 0.0

        loss_array, latent_space = args[:2]

        for in_data in dataloader:

            read_counts, rawread_counts, contigs_length, kmer, \
                kmeraug1, kmeraug2, kmeraug3, kmeraug4, kmeraug5, kmeraug6 = in_data
            if training:
                fraction_pi = args[2]
                self.optimizer.zero_grad()

            with autocast():
                if training:
                    ### augmentation by fragmentation ###
                    augmented_online, augmented_target, \
                        kmeraug_online, kmeraug_target = self.apply_augmentations(
                    rawread_counts, contigs_length, fraction_pi, [kmeraug1, kmeraug2, kmeraug3, kmeraug4, kmeraug5, kmeraug6]
                    )

                    latent, loss = \
                        self(torch.cat((augmented_online, kmeraug_online), 1), \
                            torch.cat((augmented_target, kmeraug_target), 1))

                else:
                    if self.usecuda:
                        read_counts = read_counts.cuda()
                        kmer = kmer.cuda()
                    latent, loss = \
                        self(torch.cat((read_counts, kmer), 1), \
                            torch.cat((read_counts, kmer), 1))

            loss_array.append(loss.data.item())
            latent_space.append(latent.cpu().detach().numpy())

            if training:
                # loss.backward()
                # optimizer.step()
                self.scaler.scale(loss).backward() # type: ignore
                self.scaler.step(self.optimizer) # type: ignore
                self.scaler.update() # type: ignore
                self.update_moving_average()

            epoch_loss += loss.detach().data.item()
        if training:
            self.scheduler.step() # type: ignore
        epoch_losses.extend([epoch_loss])
        self.logger.info(f'{epoch}: byol loss={epoch_loss}') # type: ignore
    
    def apply_augmentations(
        self,
        rawread_counts: torch.Tensor,
        contigs_length: torch.Tensor,
        fraction_pi: float,
        kmers: list
        ):
        """Apply data augmentations for training."""
        
        augmented_online = self.data_augment(rawread_counts, contigs_length, fraction_pi)
        augmented_target = self.data_augment(rawread_counts, contigs_length, fraction_pi)
        
        # Randomly select two k-mers for online and target networks
        kmeraug_online, kmeraug_target = random.sample(kmers, 2)

        if self.usecuda:
            # Move tensors to GPU if available
            augmented_online = augmented_online.cuda()
            augmented_target = augmented_target.cuda()
            kmeraug_online = kmeraug_online.cuda()
            kmeraug_target = kmeraug_target.cuda()
        return augmented_online, augmented_target, kmeraug_online, kmeraug_target

    # def process_batches_withpairs(self,
    #     epoch: int,
    #     dataloader,
    #     training: bool,
    #     *args):
    #     """ process batches """

    #     epoch_losses = []
    #     epoch_loss = 0.0

    #     for in_data in dataloader:

    #         pairs = in_data
    #         pairindices_1 = pairs[:, 0].to(self.read_counts.device)
    #         pairindices_2 = pairs[:, 1].to(self.read_counts.device)

    #         if training:
    #             loss_array, latent_space, fraction_pi = args
    #             self.optimizer.zero_grad()

    #         else:
    #             loss_array, latent_space = args

    #         with autocast():
    #             if training:
    #                 ### augmentation by fragmentation ###
    #                 kmeraug_online, kmeraug_target = random.sample([self.kmeraug1, \
    #                     self.kmeraug2, self.kmeraug3, self.kmeraug4, self.kmeraug5, self.kmeraug6],2)
    #                 augmented_reads1 = self.data_augment(self.rawread_counts[pairindices_1], \
    #                                     self.contigs_length[pairindices_1], fraction_pi)
    #                 augmented_reads2 = self.data_augment(self.rawread_counts[pairindices_2], \
    #                                     self.contigs_length[pairindices_2], fraction_pi)
    #                 augmented_kmers1 = kmeraug_online[pairindices_1]
    #                 augmented_kmers2 = kmeraug_target[pairindices_2]

    #                 if self.usecuda:
    #                     augmented_reads1 = augmented_reads1.cuda()
    #                     augmented_reads2 = augmented_reads2.cuda()
    #                     augmented_kmers1 = augmented_kmers1.cuda()
    #                     augmented_kmers2 = augmented_kmers2.cuda()

    #                 # rc_reads1 = torch.log(augmented_reads1.sum(axis=1))
    #                 # rc_reads2 = torch.log(augmented_reads2.sum(axis=1))
    #                 latent, loss = \
    #                     self(torch.cat((augmented_reads1, augmented_kmers1), 1), \
    #                         torch.cat((augmented_reads2, augmented_kmers2), 1), True)
    #             else:
    #                 augmented_reads = self.read_counts[pairindices_1]
    #                 augmented_kmers = self.kmer[pairindices_1]

    #                 if self.usecuda:
    #                     augmented_reads = augmented_reads.cuda()
    #                     augmented_kmers = augmented_kmers.cuda()

    #                 # rc_reads = torch.log(augmented_reads1.sum(axis=1))
    #                 input1 = torch.cat((augmented_reads, augmented_kmers), 1)
    #                 latent, loss = self(input1, input1, True)
    #         loss_array.append(loss.data.item())
    #         latent_space.append(latent.cpu().detach().numpy())

    #         if training:
    #             # loss.backward()
    #             # self.fabric.backward(loss)
    #             # optimizer.step()
    #             self.scaler.scale(loss).backward() # type: ignore
    #             self.scaler.step(self.optimizer) # type: ignore
    #             self.scaler.update() # type: ignore

    #             self.update_moving_average()

    #         epoch_loss += loss.detach().data.item()

    #     epoch_losses.extend([epoch_loss])
    #     self.logger.info(f'{epoch}: pair loss={epoch_loss}')

    # def pair_train(
    #     self,
    #     epoch,
    #     dataloader_pairtrain,
    #     dataloader_pairval,
    #     fraction_pi,
    #     loss_train,
    #     loss_val,
    # ):
    #     epoch_list = [120]
    #     if epoch in epoch_list:
    #         if self.usecuda:
    #             reads = self.read_counts[self.markercontigs].cuda()
    #             kmers = self.kmer[self.markercontigs].cuda()
    #         latent, loss = self(torch.cat((reads, kmers), 1), \
    #                 torch.cat((reads, kmers), 1))
    #         print(loss, 'loss before', epoch)
    #         np.save(self.outdir+f'latent_before_ng{epoch}',latent.cpu().detach().numpy())
    #         for epoch_pair in range(50):
    #             self.train()
    #             latent_space_train = []

    #             # initialize target network
    #             self.initialize_target_network()
    #             self.process_batches_withpairs(epoch_pair, dataloader_pairtrain, \
    #             True, loss_train, latent_space_train, fraction_pi)
    #             self.eval()
    #             latent_space_val = []

    #             with torch.no_grad():
    #                 self.process_batches_withpairs(epoch_pair, dataloader_pairval, \
    #                 False, loss_val, latent_space_val)
    #         latent, loss = self(torch.cat((reads, kmers), 1), \
    #                 torch.cat((reads, kmers), 1))
    #         print(loss, 'loss after', epoch)
    #         np.save(self.outdir+f'latent_after_ng{epoch}',latent.cpu().detach().numpy())
    #         # self.getlatent(name='after_scmg')
        

    def trainepoch(
        self,
        nepochs: int,
        dataloader_train,
        dataloader_val,
        batchsteps,
        # dataloader_pairtrain,
        # dataloader_pairval,
    ):
        """ training epoch """

        loss_train = []
        loss_val = []

        fraction_pi = self.augmentsteps[0]

        counter = 1

        # with torch.autograd.detect_anomaly():
        # detect nan occurrence (only to for loop parts)

        check_earlystop = EarlyStopper()
        # augmentation by sampling
        for epoch in range(nepochs):

            if epoch in batchsteps:

                fraction_pi = self.augmentsteps[counter]
                counter += 1

                print(fraction_pi, 'fraction pi')

            # training
            self.train()
            latent_space_train = []

            # initialize target network
            self.initialize_target_network()

            self.process_batches(epoch, dataloader_train, \
                True, loss_train, latent_space_train, fraction_pi)

            # testing
            self.eval()
            latent_space_val = []

            with torch.no_grad():
                self.process_batches(epoch, dataloader_val, \
                False, loss_val, latent_space_val)
            
            latent_space_train = []
            latent_space_val = []

        return None


    def trainmodel(
        self,
        nepochs: int = 400, #
        batchsteps: list = [],
        ):
        """ train medevol vae byol model """

        if not batchsteps:
            batchsteps = [50, 100, 150] # [1, 2, 3, 4] # [500, 1000, 2000] # [50, 100, 150, 200] # [30, 50, 70, 100], #[10, 20, 30, 45],
        batchsteps_set = sorted(set(batchsteps))

        dataloader_train = DataLoader(dataset=self.dataset_train, \
            batch_size=4096, drop_last=True, shuffle=True, \
            num_workers=self.num_workers, pin_memory=self.cuda)
        dataloader_val = DataLoader(dataset=self.dataset_val, \
            batch_size=4096, drop_last=True, shuffle=True, \
            num_workers=self.num_workers, pin_memory=self.cuda)

        # # split read mapping dataloader
        # dataloader_pairtrain = DataLoader(dataset=self.pairs_train, \
        #     batch_size= 4096, shuffle=True, drop_last=True, \
        #     num_workers=self.num_workers, pin_memory=self.cuda)
        # dataloader_pairval = DataLoader(dataset=self.pairs_val, \
        #     batch_size= 4096, shuffle=True, drop_last=True, \
        #     num_workers=self.num_workers, pin_memory=self.cuda)


        # dataloader_train, dataloader_val, \
        #     dataloader_pairtrain, dataloader_pairval = \
        #     self.fabric.setup_dataloaders(\
        #     dataloader_train, dataloader_val,\
        #     dataloader_pairtrain, dataloader_pairval)
        
        warmup_epochs = 20
        warmup_scheduler = WarmUpLR(self.optimizer, total_iters=warmup_epochs * len(dataloader_train))

        # Define the main scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100-warmup_epochs, eta_min=0)
        # self.scheduler = WarmUpThenScheduler(optimizer, warmup_scheduler, main_scheduler)

        self.trainepoch(
            nepochs, dataloader_train, dataloader_val, batchsteps_set)#, dataloader_pairtrain, dataloader_pairval)

        torch.save(self.state_dict(), self.outdir + '/byol_model.pth')

    def testmodel(self):
        """ test model """
        self.eval()

        dataloader_test = DataLoader(dataset=self.dataset_test, batch_size=4096, \
            drop_last=True, shuffle=True, num_workers=self.num_workers, pin_memory=self.cuda)
        loss_test = []
        latent_space_test = []

        with torch.no_grad():
            self.process_batches(0, dataloader_test, \
            False, loss_test, latent_space_test)

        # np.save(self.outdir + '/byol_test.npy', np.array(loss_test))

    def getlatent(self, name:str=""):
        """ get latent space after training """

        dataloader = DataLoader(dataset=self.dataset, batch_size=4096,
            shuffle=False, drop_last=False, \
            num_workers=self.num_workers, pin_memory=self.cuda)

        loss_test = []
        latent_space = []

        self.eval()
        with torch.no_grad():
            self.process_batches(0, dataloader, \
            False, loss_test, latent_space)

        np.save(self.outdir + '/latent'+name+'.npy', np.vstack(latent_space))

        return np.vstack(latent_space)

class LinearClassifier(nn.Module):
    """ Linear classifier """
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """ linear layer forward """
        return self.fc(x)

def run(abundance_matrix, outdir, contig_length, contig_names, multi_split, ncpus):
    start = time.time()
    logging.basicConfig(format='%(asctime)s - %(message)s', \
    level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S',
    filename=outdir + 'byol_training.log', filemode='w')
    logger = logging.getLogger()
    # logger.propagate = False
    logger.info('BYOL Training started')
    base_names = [''] + [f'{i}' for i in range(1, 7)]
    logger.info(f'abundance matrix shape: {abundance_matrix.shape}')
    logger.info(f'contig length shape: {contig_length.size}')
    logger.info(f'contig names shape: {contig_names.size}')
    # filter contigs with low abundances
    print(np.min(abundance_matrix.sum(axis=1)), np.max(abundance_matrix.sum(axis=1)), 'min max of total abundance')
    nonzeroindices = np.nonzero(abundance_matrix.sum(axis=1)>1.5)[0]
    print(len(np.nonzero(abundance_matrix.sum(axis=1)==0.0)[0]), 'contigs with zero total counts')
    if len(nonzeroindices) < contig_length.size:
        logger.info(f'In the dataset {contig_length.size - len(nonzeroindices)} contigs have total abundance 1.5. Removing them from binning!')
    abundance_matrix = abundance_matrix[nonzeroindices]
    contig_length = contig_length[nonzeroindices]
    contig_names = contig_names[nonzeroindices]
    print(np.min(abundance_matrix.sum(axis=1)), np.max(abundance_matrix.sum(axis=1)), 'min max of total abundance')
    logger.info('after filtering contigs with total abundance being low <1.5')
    logger.info(f'abundance matrix shape: {abundance_matrix.shape}')
    logger.info(f'contig length shape: {contig_length.size}')
    logger.info(f'contig names shape: {contig_names.size}')
    kmer_data = {}
    for counter, name in enumerate(base_names):
        keyname = "kmer" if name == '' else "kmeraug"
        if counter == 0:
            arg_name = os.path.join(outdir, f'kmer_embedding{name}.npy')
        else:
            arg_name = os.path.join(outdir, f'kmer_embedding_augment{name}.npy')
        kmerdata_tmp = np.load(arg_name, allow_pickle=True).astype(np.float32)
        kmer_data[keyname+name] = kmerdata_tmp[nonzeroindices]

    byol = BYOLmodel(abundance_matrix, kmer_data, contig_length, outdir, logger, multi_split, ncpus)
    byol.trainmodel()
    latent = byol.getlatent()
    logger.info(f"BYOL training is completed in {time.time() - start:.2f} seconds")
    np.save(os.path.join(outdir, 'contignames.npy'), np.array(contig_names))
    np.savetxt(os.path.join(outdir, 'contignames'), contig_names, fmt='%s')
    np.save(os.path.join(outdir, 'contiglength.npy'), np.array(contig_length))
    return latent, contig_length, contig_names

def main() -> None:

    """ BYOL for metagenome binning """
    start = time.time()
    parser = argparse.ArgumentParser(
        prog="mcdevol",
        description="BYOL for metagenome binning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s \
        --reads --length --names --kmer --kmeraug1 --kmeraug2 --kmeraug3 --kmeraug4 --kmeraug5 --kmeraug6 --outdir [options]",
        add_help=False,
    )

    parser.add_argument("--reads", type=str, \
        help="read coverage matrix in npz format", required=True)
    parser.add_argument("--length", type=str, \
        help="length of contigs in bp", required=True)
    parser.add_argument("--names", type=str, \
        help="ids of contigs", required=True)
    # parser.add_argument("--pairlinks", type=str, \
    #     help="provide pair links array", required=True)
    # parser.add_argument("--otuids", type=str, \
    #     help="otuids of contigs", required=True)
    parser.add_argument("--kmer", type=str, \
        help='kmer embedding', required=True)
    parser.add_argument("--kmeraug1", type=str, \
        help='kmer embedding augment 1', required=True)
    parser.add_argument("--kmeraug2", type=str, \
        help='kmer embedding augment 2', required=True)
    parser.add_argument("--kmeraug3", type=str, \
        help='kmer embedding augment 3', required=True)
    parser.add_argument("--kmeraug4", type=str, \
        help='kmer embedding augment 4', required=True)
    parser.add_argument("--kmeraug5", type=str, \
        help='kmer embedding augment 5', required=True)
    parser.add_argument("--kmeraug6", type=str, \
        help='kmer embedding augment 6', required=True)
    # parser.add_argument("--marker", type=str, \
    #     help="marker genes hit", required=True)
    parser.add_argument("--outdir", type=str, \
        help="output directory", required=True)
    parser.add_argument("--nlatent", type=int, \
        help="number of latent space")
    parser.add_argument("--cuda", \
        help="use GPU to train & cluster [False]", action="store_true")

    args = parser.parse_args()

    args.reads = np.load(args.reads, allow_pickle=True)['arr_0']
    args.length = np.load(args.length, allow_pickle=True)['arr_0']
    args.names = np.load(args.names, allow_pickle=True)['arr_0']
    # args.pairlinks = np.load(args.pairlinks, allow_pickle='True')
    args.kmer = np.load(args.kmer, allow_pickle=True).astype(np.float32)
    args.kmeraug1 = np.load(args.kmeraug1, allow_pickle=True).astype(np.float32)
    args.kmeraug2 = np.load(args.kmeraug2, allow_pickle=True).astype(np.float32)
    args.kmeraug3 = np.load(args.kmeraug3, allow_pickle=True).astype(np.float32)
    args.kmeraug4 = np.load(args.kmeraug4, allow_pickle=True).astype(np.float32)
    args.kmeraug5 = np.load(args.kmeraug5, allow_pickle=True).astype(np.float32)
    args.kmeraug6 = np.load(args.kmeraug6, allow_pickle=True).astype(np.float32)


    base_names = ['kmer'] + [f'kmeraug{i}' for i in range(1, 7)]
    kmer_data = {}
    for name in base_names:
        kmer_data[name] = getattr(args, name)
    
    # scale kmer input
    # first attempt * 512 1-3
    # second attempt * 100 1-3
    # third attempt * 10 1-3
    # four attempt * 150 1-3
    # five attempt * 200 1-3
    # args.kmer = args.kmer * 200
    # args.kmeraug1 = args.kmeraug1 * 200
    # args.kmeraug2 = args.kmeraug2 * 200

    # args.marker = pd.read_csv(args.marker, header=None, sep='\t')
    # names_indices = {name: i for i, name in enumerate(args.names)}
    # args.marker[2] = args.marker[0].map(names_indices)
    # del names_indices
    # # remove contigs having scmg but shorter than threshold length
    # args.marker = args.marker.dropna(subset=[2])
    # args.marker[2] = args.marker[2].astype('int')
    # args.marker = dict(args.marker.groupby(1)[2].apply(list))

    args.outdir = os.path.join(args.outdir, '')

    try:
        if not os.path.exists(args.outdir):
            print('create output folder')
            os.makedirs(args.outdir)
    except RuntimeError as e:
        print(f'output directory already exist. Using it {e}')

    logging.basicConfig(format='%(asctime)s - %(message)s', \
    level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S',
    filename=args.outdir + '/byol_training.log', filemode='w')
    args.logger = logging.getLogger()

    byol = BYOLmodel(args.reads, kmer_data, args.contig_length, args.outdir, args.logger, False)

    # total_params = sum(p.numel() for p in byol.parameters() if p.requires_grad)

    byol.trainmodel()
    # byol.testmodel()
    latent = byol.getlatent()
    print(f"BYOL training is completed in {time.time() - start:.2f} seconds")

    args.logger.info(f'{time.time()-start:.2f}, seconds to complete')
if __name__ == "__main__" :
    main()