import selfies as sf
import rdkit.Chem as Chem

def smiles_from_sflabels(labels, vocab_itos):
    sfs = sf.encoding_to_selfies(
        labels, vocab_itos=vocab_itos, enc_type="label"
    )
    m = Chem.MolFromSmiles(sf.decoder(sfs))

    if m is None:
        return ''
    else:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)

