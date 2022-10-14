import torch
import torch.nn as nn
import torch.nn.function as F

from torch.utils.data import Dataset

import selfies as sf

class selfies_dataset(Dataset):
    def __init__(self, sf_list, len_molecule = None):
        self.sf_list = sf_list
        self.len_molecule = max(sf.len_selfies(s) for s in self.sf_list) if len_molecule is None else len_molecule

        # create the alphabet
        self.alphabet = sf.get_alphabet_from_selfies(sf_list)
        self.alphabet = list(sorted(self.alphabet))
        self.alphabet.insert(0, "[nop]")

        # create the conversion vocab
        self.vocab_stoi = {s, i for i, s in enumerate(self.alphabet)}
        self.vocab_itos = {i, s for s, i in self.vocab_stoi.items()}

    def __len__(self):
        return len(sf_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = sf.selfies_to_encoding(
            self.sf_list[idx],
            vocab_stoi = self.vocab_stoi,
            pad_to_len = self.len_molecule,
            enc_type = "label"
        )

        target_label = torch.tensor(label).to(torch.long)

        label.insert(0, self.vocab_stoi["[nop]"])
        input_label = torch.tensor(label).to(torch.long)

        return input_label, target_label

class rl_dataset(Dataset):
    def __init__(self, sequence, log_prob, scores):
        self.seq = sequence
        self.log_prob = log_prob
        self.scores = scores

    def __len__(self):
        return len(self.log_prob)

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        seq = torch.tensor(self.seq[idx, :]).to(torch.long)
        log_prob = torch.tensor(self.log_prob[idx, :]).to(torch.float)

        return seq, log_prob
