import copy
import multiprocessing
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .dataset import selfies_dataset, rl_dataset
from .utils import smiles_from_sflabels

from torch.utils.data import DataLoader

import pytorch_lightning as pl

class LanguageModel(nn.Module):
    def __init__(
        self, 
        len_alphabet: int, 
        len_molecule: int, 
        embedding_dim :int, 
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.len_alphabet = len_alphabet
        self.len_molecule = len_molecule
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.len_alphabet, self.embedding_dim)
        self.rnn = nn.GRU(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True
        )
        self.fc = nn.Linear(self.hidden_dim, self.len_alphabet)
    
    def forward(self, x, h = None):
        x = self.embedding(x)
        x, h = self.rnn(x, h)
        x = self.fc(x)
        return x, h

    def sample(self, num_samples, temp: float = 1.0):
        # sample new sequences, note that the pad variable is 
        # assumed to be indexed 0
        start_tokens = torch.zeros(num_samples).long()
        
        # initialize
        h = None
        log_prob = torch.zeros(num_samples)
        sequence = []

        for _ in range(self.len_molecule):
            x, h = self.forward(x, h)
            x = x / temp
            prob = F.softmax(x, dim=-1)
            x = torch.multinomial(prob, num_samples=1).view(-1)
            
            # save log probability and the sequence
            log_prob += F.log_softmax(x, dim=-1)
            sequence.append(x.view(-1, 1))
        sequence = torch.cat(sequence, 1)

        # remove duplicate sequences
        sequence, unique_idx = np.unique(sequence.numpy(), axis=0, return_index=True)
        sequence = torch.tensor(sequence).to(torch.long)
        log_prob = log_prob[unique_idx]
        return sequence, log_prob


class Prior(pl.LightningModule):
    def __init__(self, prior_network: nn.Module):
        super().__init__()
        self.prior_network = prior_network
        self.nll = nn.NLLLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.prior_network(x)    # logsoftmax already applied
        loss = self.nll(F.log_softmax(y_hat, dim=-1), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.prior_network(x)    # logsoftmax already applied
        loss = self.nll(F.log_softmax(y_hat, dim=-1), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class Agent(pl.LightningModule):
    def __init__(
        self, 
        scoring_function: Callable, 
        sigma: float, 
        num_samples: int,
        vocab_itos, 
        prior_network: nn.Module, 
        agent_network: nn.Module = None,
        num_workers = None,
    ):
        super().__init__()
        self.prior_network = prior_network
        for param in self.prior_network.parameters():
            param.requires_grad = False     # freeze the prior
        
        self.scoring_function = scoring_function
        self.sigma = sigma
        self.num_samples = num_samples
        self.nll = nn.NLLLoss()
        self.vocab_itos = vocab_itos
        self.agent_network = copy.deepcopy(prior_network) if agent_network is None else agent_network
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers

    def generate_dataloader(self):
        # generate a dataloader by sampling the agent
        seq, log_prob = self.prior_network.sample(self.num_samples)
        
        with multiprocessing.Pool(self.num_workers) as pool:
            smi_list = pool.map(partial(smiles_from_sflabels, self.vocab_itos), seq)
            scores = pool.map(self.scoring_function, smi_list)
        dl = DataLoader(rl_dataset(seq, log_prob, scores))
        return dl

    def training_step(self, batch, batch_idx):
        x, log_prob, score = batch
        y_hat, _ = self.prior_network(x)
        prior_likelihood = self.nll(F.log_softmax(y_hat, dim=-1), y)
        aug_likelihood = prior_likelihood + self.sigma * score

        loss = torch.pow(aug_likelihood - log_prob, 2)

        loss_p = -(1 / log_prob).mean()
        loss += 5e3 * loss_p
        return loss

    # def validation_step(self, batch, batch_idx):
    #     return
