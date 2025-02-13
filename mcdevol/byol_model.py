#!/usr/bin/env python
""" run byol training """

import math
import os
import time
import copy
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from numpy import random as np_random
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import logging
from torch.cuda.amp import autocast, GradScaler
import wandb
wandb.login(key="04479eb61e281cd04fb842aa4636701a7740bc8c")
wandb.init(project="gradient-monitoring bathypelagic pooled with no amp")

def compute_weight_norm(parameters):
    total_norm = 0.0
    for param in parameters:
        param_norm = param.detach().norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def compute_grad_norm(parameters):
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.detach().norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def normalize_counts(counts):
    """ normalize count by mean division """
    if isinstance(counts, np.ndarray):
        counts_norm = counts / counts.mean(axis=1, keepdims=True)
    elif isinstance(counts, torch.Tensor):
        counts_norm = counts / counts.mean(dim=1, keepdim=True)
    else:
        raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
    
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

def get_fraction_pi():
    eps = torch.finfo(torch.float32).eps
    # sample = torch.rand(1).item() * (1.0 - 2 * eps) + eps
    # Sample from (0.3, 0.9)
    sample = torch.rand(1).item() * (0.9 - 0.3 - 2 * eps) + (0.3 + eps)
    return sample

def split_dataset(dataset, flag_test):
    """ split dataset into training and validation. Also, test set (if flag_test is true)"""
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    remaining = total_size - train_size

    if flag_test:
        val_size = remaining // 2
        test_size = remaining - val_size
        return random_split(dataset, [train_size, val_size, test_size])
    else:
        val_size = remaining
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

def target_update_moving_average(
    ema_updater,
    online_encode,
    online_project,
    target_encode,
    target_project
):
    """
    Update target network parameters using EMA.
    Args:
        ema_updater (EMAUpdater): EMA update utility.
        online_encode (torch.nn.Module): Online encoder network.
        online_project (torch.nn.Module): Online projector network.
        target_encode (torch.nn.Module): Target encoder network.
        target_project (torch.nn.Module): Target projector network.
    """

    for online_params, target_params in zip(
        online_encode.parameters(), target_encode.parameters()
    ):
        target_params.data = ema_updater.update_average(target_params.data, online_params.data)

    for online_params, target_params in zip(
        online_project.parameters(), target_project.parameters()
    ):
        target_params.data = ema_updater.update_average(target_params.data, online_params.data)

    return None

class BYOLmodel(nn.Module):
    """ train BYOL model """
    def __init__(
        self,
        abundance_matrix: np.ndarray,
        kmer_data: dict,
        len_data: dict,
        contig_length: np.ndarray,
        outdir: str,
        logger: object,
        multi_split: bool,
        ncpus: int = 8,
        readlength: int = 250,
        lrate: float = 0.1,
        n_fragments: int = 6,
        seed: int = 0,
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
        self.read_length = readlength
        self.lrate = lrate
        self.n_fragments = n_fragments
        # Model architecture parameters
        self.nhidden = 1024
        self.nlatent = 512
        self.dropout = 0.1
        projection_size = 256
        projection_hidden_size = 4096
        self.batch_size = 4096
 
        if int(self.ncontigs * 0.8) < self.batch_size:
            self.batch_size = int(self.ncontigs * 0.2)

        # abundance matrix usually is a coverage, ie., the number of bases mapped to a contig, divided by the length of the contig
        # Hence, read_coverage is the correct nomenclature for input data
        self.read_coverage = torch.from_numpy(normalize_counts(abundance_matrix))

        def getreadcounts(coverage_matrix, contig_length):
            return (coverage_matrix * contig_length[:, np.newaxis]) / self.read_length

        read_counts = getreadcounts(abundance_matrix, contig_length)
        self.read_counts = torch.from_numpy(read_counts.astype(np.float32))
        self.contigs_length = torch.from_numpy(contig_length)
        
        # get kmer data
        for key, value in kmer_data.items():
            setattr(self, key, torch.from_numpy(value))
        
        for key, value in len_data.items():
            setattr(self, key, torch.from_numpy(value))

        kmeraug_list = [getattr(self, f'kmeraug{i}') for i in range(1, self.n_fragments + 1)]
        len_list = [getattr(self, f'len{i}') for i in range(1, self.n_fragments + 1)]

        self.indim = self.nsamples + self.kmeraug1.shape[1]
        self.dataset = TensorDataset(self.read_coverage, self.read_counts,\
            self.contigs_length, self.kmer, *kmeraug_list, *len_list)
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
        self.target_encoder = None
        self.target_projector = None
        moving_average_decay = 0.996
        self.target_ema_updater = EMA(moving_average_decay)

        if self.usecuda:
            self.cuda()
            print('Using device:', self.device)

            #Additional Info when using cuda
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

        trainable_params = list(self.online_encoder.parameters()) + \
                   list(self.online_projector.parameters()) + \
                   list(self.online_predictor.parameters())

        # self.optimizer = SGD(trainable_params, lr=0.03, momentum=0.9, weight_decay=1.5e-6)

        params_with_decay = []
        params_without_decay = []

        # Loop over the online encoder, projector, and predictor parameters
        for module in [self.online_encoder, self.online_projector, self.online_predictor]:
            for name, param in module.named_parameters():
                if 'bias' in name or 'bn' in name:  # Exclude biases and batch norm parameters from weight decay
                    params_without_decay.append(param)
                else:
                    params_with_decay.append(param)
        
        self.optimizer = SGD([
            {'params': params_with_decay, 'weight_decay': 1.5e-6},  # Apply weight decay to these
            {'params': params_without_decay, 'weight_decay': 0}  # No weight decay for these
        ], lr=self.lrate, momentum=0.9)

        mean_coverage = np.mean(abundance_matrix.sum(axis=1))
        self.logger.info(f"mean coverage:{mean_coverage}")
        if mean_coverage > 20:
            self.unscale = True
        else:
            # Gradients are too low for clipping without unscaling. Scaling gradients up makes clipping more effective.
            self.unscale = False

    def data_augment(
        self, 
        rcounts: torch.Tensor,
        contigs_length: torch.Tensor,
        fraction_pi: float):
        """ augment read counts """
        rcounts_sampled = rcounts.clone().detach()

        rcounts_sampled = drawsample_frombinomial(rcounts_sampled, fraction_pi)
        zeroindices = (rcounts_sampled.sum(axis=1)==0.0).nonzero()
        print(zeroindices, 'zeroindices', flush=True)
        if len(zeroindices) > 0:
            print(contigs_length[zeroindices], 'contig length', flush=True)
            print(rcounts[zeroindices], 'rcounts sampled', flush=True)
            print(rcounts_sampled[zeroindices], 'rcounts sampled', flush=True)
            raise ValueError("sample_counts is zero counts across all samples")
        return rcounts_sampled

    def initialize_target_network(self):
        """ initialize target network """
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Move to the same device as the online networks
        self.target_encoder = self.target_encoder.to(self.device)
        self.target_projector = self.target_projector.to(self.device)

        # Freeze parameters for target_encoder and target_projector
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def update_moving_average(self):
        """ update target network by moving average """

        target_update_moving_average(
            self.target_ema_updater, 
            self.online_encoder,
            self.online_projector,
            self.target_encoder,
            self.target_projector
        )

    def forward(self, x, xt):
        """ forward BYOL """
        latent = self.online_encoder(x)
        z1_online = self.online_predictor(self.online_projector(latent))
        z2_online = self.online_predictor(self.online_projector(self.online_encoder(xt)))
        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(xt)) # type: ignore
            z2_target =  self.target_projector(self.target_encoder(x)) # type: ignore

        byol_loss = self.compute_loss(z1_online, z1_target.detach()) + \
            self.compute_loss(z2_online, z2_target.detach()) # to symmetrize the loss

        return latent, byol_loss

    def compute_loss(self, z1, z2):
        """ loss for BYOL """
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        return torch.mean(2 - 2 * (z1 * z2).sum(dim=-1))


    def multinomialsampling(self, read_counts, fraglength_coverage, eps=1e-8):
        # Dirichlet sampling

        probabilities = torch.distributions.Dirichlet(read_counts + 1.0 + eps).sample()
        assert torch.all(probabilities.sum(dim=1, keepdim=True) > 0), "dirichlet sampling"
        # probabilities = torch.clamp(probabilities, min=eps)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        # print(torch.min(probabilities), torch.max(probabilities), 'min max probabilites from Dirichlet', flush=True)

        # Compute augmented counts
        total_counts = torch.sum(read_counts, dim=1).float()
        # print(total_counts[0], torch.sum(total_counts[0] * probabilities[0]), 'total and Dirichlet sampling', flush=True)
        augmented_total_counts = total_counts * fraglength_coverage

        assert torch.all(fraglength_coverage > 0), "Fraglength coverage contains zero or negative values!"
        assert torch.all(total_counts > 0), "Total counts contain zero!"

        # Scale probabilities
        scaled_probs = augmented_total_counts.unsqueeze(-1) * probabilities
        # print(torch.min(scaled_probs), torch.max(scaled_probs), 'min max scaled_probs', flush=True)
        # scaled_probs = torch.clamp(scaled_probs, min=eps, max=1e3)  # Clamp small and large values for stability

        if torch.any(torch.isnan(scaled_probs)) or torch.any(scaled_probs <= 0):
            raise ValueError("Invalid values in scaled_probs!")
    
        # Poisson sampling
        multinomial_samples = torch.clamp(torch.poisson(scaled_probs), max=torch.max(read_counts))

        # Resample if zero count vectors are obtained
        sampled_totals = multinomial_samples.sum(dim=1, keepdim=True)
        zeroindices = (sampled_totals == 0).squeeze()

        max_attempts = 10
        for attempt in range(max_attempts):
            sampled_totals = multinomial_samples.sum(dim=1, keepdim=True)
            zeroindices = (sampled_totals == 0).squeeze()
            if not zeroindices.any():
                break
            print(f"Resampling attempt {attempt + 1} for indices: {zeroindices.nonzero().squeeze()}")
            multinomial_samples[zeroindices] = torch.poisson(scaled_probs[zeroindices])
        else:
            raise RuntimeError(f"Failed to rebalance multinomial_samples after {max_attempts} attempts")

        return multinomial_samples.to(torch.float32)

    def apply_augmentations(
        self,
        read_counts: torch.Tensor,
        contigs_length: torch.Tensor,
        kmers: list,
        lengths: list,
    ):
        """Apply data augmentations for training."""
        
        # Randomly select two k-mers for online and target networks
        device = read_counts.device
        contigs_length = contigs_length.to(device)
        index1, index2 = torch.randperm(self.n_fragments)[:2].tolist()
        kmeraug_online, kmeraug_target = kmers[index1].to(device), kmers[index2].to(device)
        length_online, length_target = lengths[index1].to(device), lengths[index2].to(device)
        # augmented_online = self.data_augment(read_counts, contigs_length, fraction_pi)
        # augmented_target = self.data_augment(read_counts, contigs_length, fraction_pi)
        # print(contigs_length, 'complete length', flush=True)
        # print(length_online, 'length online', flush=True)
        # print(length_target, 'length target', flush=True)
        fraglength_coverage_online = (length_online / contigs_length).to(torch.float32)
        fraglength_coverage_target = (length_target / contigs_length).to(torch.float32)

        augmented_online = self.multinomialsampling(read_counts, fraglength_coverage_online)
        augmented_target = self.multinomialsampling(read_counts, fraglength_coverage_target)

        # if self.usecuda:
        #     # Move tensors to GPU if available
        #     augmented_online = augmented_online.cuda()
        #     augmented_target = augmented_target.cuda()
        #     kmeraug_online = kmeraug_online.cuda()
        #     kmeraug_target = kmeraug_target.cuda()
    
        return augmented_online.to(device), augmented_target.to(device), kmeraug_online.to(device), kmeraug_target.to(device)
    
    def getcoverage(self, aug_read_counts, contigs_length):
        return (aug_read_counts * self.read_length) / contigs_length.unsqueeze(1)
    
    def adaptive_gradient_clipping(self, parameters, alpha=0.01, eps=1e-6):
    
        for param in parameters:
            if param.grad is not None and param.requires_grad:
                # Compute the norm of the parameter weights
                weight_norm = torch.norm(param, p=2)
                
                # Compute the norm of the gradients
                grad_norm = torch.norm(param.grad, p=2)
                
                # Compute the clipping threshold dynamically
                max_grad_norm = alpha * (weight_norm + eps)
                
                # Clip gradients if they exceed the threshold
                if grad_norm > max_grad_norm:
                    param.grad.data.mul_(max_grad_norm / (grad_norm + eps))
      
    def process_batches(self,
        epoch: int,
        dataloader,
        training: bool,
        *args):
        """ process batches """

        epoch_loss = 0.0

        loss_array, latent_space = args[:2]

        for in_data in dataloader:
            in_data = [d.to(self.device) for d in in_data]
            read_coverage, read_counts, contigs_length, kmer, \
                *augdata = in_data
            kmer_augs, len_augs = augdata[:self.n_fragments], augdata[self.n_fragments:]

            if training:
                self.optimizer.zero_grad()

                with autocast():
                    augmented_online, augmented_target, \
                        kmeraug_online, kmeraug_target = self.apply_augmentations(
                        read_counts, contigs_length,
                        kmer_augs,
                        len_augs
                    )

                    augmented_online_cov = normalize_counts(self.getcoverage(augmented_online, contigs_length))
                    augmented_target_cov = normalize_counts(self.getcoverage(augmented_target, contigs_length))

                    latent, loss = \
                        self(torch.cat((augmented_online_cov, kmeraug_online), 1), \
                            torch.cat((augmented_target_cov, kmeraug_target), 1))

                # loss.backward()
                for param_group in self.optimizer.param_groups:
                    if param_group['weight_decay'] == 0: 
                        param_group['lr'] = self.lrate

                self.scaler.scale(loss).backward()
                
                grad_norm_encoder = compute_grad_norm(self.online_encoder.parameters())
                weight_norm_encoder = compute_weight_norm(self.online_encoder.parameters())
                grad_norm_projector = compute_grad_norm(self.online_projector.parameters())
                weight_norm_projector = compute_weight_norm(self.online_projector.parameters())
                grad_norm_predictor = compute_grad_norm(self.online_predictor.parameters())
                weight_norm_predictor = compute_weight_norm(self.online_predictor.parameters())

                wandb.log({"Encoder Gradient Norm": grad_norm_encoder, "Weight Norm": weight_norm_encoder})
                wandb.log({"Projector Gradient Norm": grad_norm_projector, "Weight Norm": weight_norm_projector})
                wandb.log({"Predictor Gradient Norm": grad_norm_predictor, "Weight Norm": weight_norm_predictor})
                wandb.log({"Scaler scale": self.scaler.get_scale()})
                if self.unscale:
                    self.scaler.unscale_(self.optimizer)

                grad_norm_encoder = compute_grad_norm(self.online_encoder.parameters())
                weight_norm_encoder = compute_weight_norm(self.online_encoder.parameters())
                grad_norm_projector = compute_grad_norm(self.online_projector.parameters())
                weight_norm_projector = compute_weight_norm(self.online_projector.parameters())
                grad_norm_predictor = compute_grad_norm(self.online_predictor.parameters())
                weight_norm_predictor = compute_weight_norm(self.online_predictor.parameters())

                wandb.log({"Unscaled Encoder Gradient Norm": grad_norm_encoder, "Weight Norm": weight_norm_encoder})
                wandb.log({"Unscaled Projector Gradient Norm": grad_norm_projector, "Weight Norm": weight_norm_projector})
                wandb.log({"Unscaled Predictor Gradient Norm": grad_norm_predictor, "Weight Norm": weight_norm_predictor})
                self.adaptive_gradient_clipping(self.online_encoder.parameters())
                self.adaptive_gradient_clipping(self.online_projector.parameters())
                self.adaptive_gradient_clipping(self.online_predictor.parameters())
                
                self.scaler.step(self.optimizer)
                # self.optimizer.step()
                self.scaler.update()
                self.update_moving_average()

            else:
                with torch.no_grad():
                    latent, loss = \
                        self(torch.cat((read_coverage, kmer), 1), \
                            torch.cat((read_coverage, kmer), 1))

            loss_array.append(loss.item())
            latent_space.append(latent.cpu().detach().numpy())
            epoch_loss += loss.detach().item()

    def get_scale_value(self, dataloader_train):
        grad_norms = []
        for in_data in dataloader_train:
            in_data = [d.to(self.device) for d in in_data]
            _, read_counts, contigs_length, _, \
                *augdata = in_data
            kmer_augs, len_augs = augdata[:self.n_fragments], augdata[self.n_fragments:]

            self.optimizer.zero_grad()

            augmented_online, augmented_target, \
                kmeraug_online, kmeraug_target = self.apply_augmentations(
                read_counts, contigs_length,
                kmer_augs,
                len_augs
            )

            augmented_online_cov = normalize_counts(self.getcoverage(augmented_online, contigs_length))
            augmented_target_cov = normalize_counts(self.getcoverage(augmented_target, contigs_length))

            _, loss = \
                self(torch.cat((augmented_online_cov, kmeraug_online), 1), \
                    torch.cat((augmented_target_cov, kmeraug_target), 1))

            loss.backward()

            all_params = list(self.online_encoder.parameters()) + \
                        list(self.online_predictor.parameters()) + \
                        list(self.online_predictor.parameters())

            batch_grad_norms = [torch.norm(p.grad).item() for p in all_params if p.grad is not None]

            grad_norms.extend(batch_grad_norms)

        if not grad_norms:  # No gradients found
            return 2.0
        
        percentile_norm = np.percentile(grad_norms, 5)

        # Handle division by zero if percentile is zero
        if percentile_norm <= 1e-9:
            return 2.0  
    
        underflow_threshold = 6e-5
        init_scale = (underflow_threshold / percentile_norm) * 2

        return init_scale

    def trainepoch(
        self,
        nepochs: int,
        dataloader_train,
        dataloader_val,
    ):
        """ training epoch """

        # with torch.autograd.detect_anomaly():
        # detect nan occurrence (only to for loop parts)
        init_scale_value = self.get_scale_value(dataloader_train)
        self.logger.info(f'init scale value {init_scale_value}')
        print(init_scale_value, 'init scale value', flush=True)
        self.scaler = GradScaler(init_scale=init_scale_value)
        # self.scaler = GradScaler(init_scale=2.0)
        check_earlystop = EarlyStopper()
        for epoch in range(nepochs):

            loss_train = []
            loss_val = []

            # fraction_pi = get_fraction_pi()

            # training
            self.train()
            latent_space_train = []

            self.process_batches(epoch, dataloader_train, \
                True, loss_train, latent_space_train)

            # validation
            self.eval()
            latent_space_val = []

            with torch.no_grad():
                self.process_batches(epoch, dataloader_val, \
                False, loss_val, latent_space_val)

            avg_loss_train = sum(loss_train) / len(loss_train) if loss_train else float('inf')
            avg_loss_val = sum(loss_val) / len(loss_val) if loss_val else float('inf')
            
            self.scheduler.step()
            
            self.logger.info(f'Epoch {epoch + 1}/{nepochs} - Learning Rate = {self.get_lr()}, Training Loss: {avg_loss_train:.4f}, Validation Loss: {avg_loss_val:.4f}')
            
            latent_space_train.clear()
            latent_space_val.clear()
            if check_earlystop.early_stop(avg_loss_val):  # Use the most recent validation loss
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        return None

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def trainmodel(
        self,
        nepochs: int = 400,
    ):
        """ train medevol vae byol model """

        dataloader_train = DataLoader(dataset=self.dataset_train, \
            batch_size = self.batch_size, drop_last=True, shuffle=True, \
            num_workers = self.num_workers, pin_memory=self.cuda)
        dataloader_val = DataLoader(dataset=self.dataset_val, \
            batch_size = self.batch_size, drop_last=False, shuffle=True, \
            num_workers = self.num_workers, pin_memory=self.cuda)

        warmup_epochs = 20

        # Linear warm-up scheduler
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=nepochs - warmup_epochs)

        # Combine schedulers
        self.scheduler = SequentialLR(self.optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs])


        self.initialize_target_network()
        self.logger.info("Starting training...")
        self.trainepoch(
            nepochs, dataloader_train, dataloader_val)

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': nepochs
        }, self.outdir + '/byol_model.pth')
    
        return None

    def testmodel(self):
        """ test model """
        self.eval()

        dataloader_test = DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            drop_last=True, shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.cuda
        )
        loss_test = []
        latent_space_test = []

        with torch.no_grad():
            self.process_batches(0, dataloader_test, \
            False, loss_test, latent_space_test)

        # np.save(self.outdir + '/byol_test.npy', np.array(loss_test))

    def getlatent(self, name:str=""):
        """ get latent space after training """

        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
            shuffle=False, drop_last=False, \
            num_workers=self.num_workers, pin_memory=self.cuda)

        loss_test = []
        latent_space = []

        self.eval()
        with torch.no_grad():
            self.process_batches(0, dataloader, \
            False, loss_test, latent_space)

        np.save(self.outdir + '/latent.npy', np.vstack(latent_space))

        return np.vstack(latent_space)

class LinearClassifier(nn.Module):
    """ Linear classifier """
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """ linear layer forward """
        return self.fc(x)

def run(
    abundance_matrix,
    outdir,
    contig_length,
    contig_names,
    multi_split,
    ncpus,
    readlength,
    lr,
    n_fragments,
):
    start = time.time()
    logging.basicConfig(format='%(asctime)s - %(message)s', \
    level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S',
    filename=outdir + 'byol_training.log', filemode='w')
    logger = logging.getLogger()
    logger.info('BYOL Training started')
    base_names = [''] + [f'{i}' for i in range(1, n_fragments+1)]
    logger.info(f'abundance matrix shape: {abundance_matrix.shape}')
    logger.info(f'contig length shape: {contig_length.size}')
    logger.info(f'contig names shape: {contig_names.size}')

    # filter contigs with low abundances
    nonzeroindices = np.nonzero(abundance_matrix.sum(axis=1)>1.0)[0]
    print(len(np.nonzero(abundance_matrix.sum(axis=1)<1.0)[0]), 'contigs with < 1.0 total counts')
    if len(nonzeroindices) < contig_length.size:
        logger.info(f'In the dataset {contig_length.size - len(nonzeroindices)} contigs have total abundance < 1.0. Removing them from binning!')
        abundance_matrix = abundance_matrix[nonzeroindices]
        contig_length = contig_length[nonzeroindices]
        contig_names = contig_names[nonzeroindices]
    logger.info('after filtering contigs with total abundance being low zero or <1.0')
    logger.info(f'abundance matrix shape: {abundance_matrix.shape}')
    logger.info(f'contig length shape: {contig_length.size}')
    logger.info(f'contig names shape: {contig_names.size}')

    kmer_data = {}
    augmented_length = {}
    for counter, name in enumerate(base_names):
        keyname = "kmer" if name == '' else "kmeraug"
        lenname = "len"
        if counter == 0:
            filename = os.path.join(outdir, f'kmer_embedding{name}.npy')
        else:
            filename = os.path.join(outdir, f'kmer_embedding_augment{name}.npy')
            lenfile = os.path.join(outdir, f'augmented_contiglength{name}.npy')
            lendata = np.load(lenfile, allow_pickle=True).astype(np.float32)
            augmented_length[lenname+name] = lendata[nonzeroindices]
        kmerdata_tmp = np.load(filename, allow_pickle=True).astype(np.float32)
        kmer_data[keyname+name] = kmerdata_tmp[nonzeroindices]

    byol = BYOLmodel(abundance_matrix, kmer_data, augmented_length, contig_length, outdir, logger, multi_split, ncpus, readlength, lrate=lr)
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
        --reads --length --names --kmer --kmeraug1 --kmeraug2 --kmeraug3 --kmeraug4 --kmeraug5 --kmeraug6 --outdir [options] --lrate [options, 3e-2]",
        add_help=False,
    )

    parser.add_argument("--reads", type=str, \
        help="read coverage matrix in npz format", required=True)
    parser.add_argument("--length", type=str, \
        help="length of contigs in bp", required=True)
    parser.add_argument("--names", type=str, \
        help="ids of contigs", required=True)
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
    parser.add_argument("--len1", type=str, \
        help='length of fragmented contigs 1', required=True)
    parser.add_argument("--len2", type=str, \
        help='length of fragmented contigs 2', required=True)
    parser.add_argument("--len3", type=str, \
        help='length of fragmented contigs 3', required=True)
    parser.add_argument("--len4", type=str, \
        help='length of fragmented contigs 4', required=True)
    parser.add_argument("--len5", type=str, \
        help='length of fragmented contigs 5', required=True)
    parser.add_argument("--len6", type=str, \
        help='length of fragmented contigs 6', required=True)
    parser.add_argument("--outdir", type=str, \
        help="output directory", required=True)
    parser.add_argument("--nlatent", type=int, \
        help="Dimension of latent space")
    parser.add_argument("--lrate", type=int, \
        help="learing rate", default=3e-2)
    parser.add_argument("--cuda", \
        help="use GPU to train & cluster [False]", action="store_true")

    args = parser.parse_args()

    args.reads = np.load(args.reads, allow_pickle=True)['arr_0']
    args.length = np.load(args.length, allow_pickle=True)['arr_0']
    args.names = np.load(args.names, allow_pickle=True)['arr_0']
    args.kmer = np.load(args.kmer, allow_pickle=True).astype(np.float32)
    args.kmeraug1 = np.load(args.kmeraug1, allow_pickle=True).astype(np.float32)
    args.kmeraug2 = np.load(args.kmeraug2, allow_pickle=True).astype(np.float32)
    args.kmeraug3 = np.load(args.kmeraug3, allow_pickle=True).astype(np.float32)
    args.kmeraug4 = np.load(args.kmeraug4, allow_pickle=True).astype(np.float32)
    args.kmeraug5 = np.load(args.kmeraug5, allow_pickle=True).astype(np.float32)
    args.kmeraug6 = np.load(args.kmeraug6, allow_pickle=True).astype(np.float32)

    args.len1 = np.load(args.len1, allow_pickle=True).astype(np.float32)
    args.len2 = np.load(args.len2, allow_pickle=True).astype(np.float32)
    args.len3 = np.load(args.len3, allow_pickle=True).astype(np.float32)
    args.len4 = np.load(args.len4, allow_pickle=True).astype(np.float32)
    args.len5 = np.load(args.len5, allow_pickle=True).astype(np.float32)
    args.len6 = np.load(args.len6, allow_pickle=True).astype(np.float32)

    base_names = ['kmer'] + [f'kmeraug{i}' for i in range(1, 7)]
    kmer_data = {}
    for name in base_names:
        kmer_data[name] = getattr(args, name)
    len_data = {}
    base_names = [f'kmeraug{i}' for i in range(1, 7)]
    for name in base_names:
        len_data[name] = getattr(args, name)
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

    byol = BYOLmodel(args.reads, kmer_data, len_data, args.contig_length, args.outdir, args.logger, False, 8, 250, lrate=args.lrate)

    byol.trainmodel()
    # byol.testmodel()
    latent = byol.getlatent()
    print(f"BYOL training is completed in {time.time() - start:.2f} seconds")

    args.logger.info(f'{time.time()-start:.2f}, seconds to complete')
if __name__ == "__main__" :
    main()