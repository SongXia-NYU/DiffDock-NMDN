# Normalized Mixture Density Network (NMDN)

It is the official implementation of the NMDN model published at [REPLACE_ME]().

![](./model.png)

# Environment Setup
WIP

# Run NMDN model
To run prediction, you need the protein structure `PROTEIN.pdb` and docked ligand poses `LIG1.sdf`, `LIG2.sdf`, ...

```bash
python predict.py --prot PROTEIN.pdb --ligs LIG1.sdf LIG2.sdf ...
```

You will get prediction of the NMDN score and pKd score for each protein-ligand pairs. If you only want to predict the NMDN score, you can add the `--nmdn_only` argument.

```bash
python predict.py --nmdn_only --prot PROTEIN.pdb --ligs LIG1.sdf LIG2.sdf ...
```
