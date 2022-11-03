import os, sys
sys.path.append('..')

from self_reinvent.network import LanguageModel, Prior, Agent
from self_reinvent.dataset import selfies_dataset

import pandas as pd
import selfies as sf

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


if __name__ == '__main__':
    print('Pretraining network')
    df = pd.read_csv('starting_smiles.csv')

    df['selfies'] = df['smiles'].apply(sf.encoder)

    sf_dataset = selfies_dataset(df['selfies'].tolist())
    train_portion = int(len(sf_dataset) * 0.8)
    sf_train, sf_valid = random_split(sf_dataset,
        lengths = [train_portion, len(sf_dataset) - train_portion])
    trainloader = DataLoader(sf_train, batch_size=32, shuffle=True)
    validloader = DataLoader(sf_valid, batch_size=32)

    net = LanguageModel(
        len_alphabet = len(sf_dataset.alphabet),
        len_molecule = sf_dataset.len_molecule,
        embedding_dim = 128,
        hidden_dim = 512,
        num_layers = 3
    )
    import pdb; pdb.set_trace()
    prior = Prior(net)
    trainer = pl.Trainer(max_epochs = 1)
    trainer.fit(prior, trainloader, validloader)

    import pdb; pdb.set_trace()
    


