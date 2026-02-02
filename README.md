# Normalized Mixture Density Network (NMDN)

It is the official implementation of the NMDN model published in the article [Normalized Protein-Ligand Distance Likelihood Score for End-to-end Blind Docking and Virtual Screening]().

![](./model.png)

# Environment Setup

```bash
# change the cuda version if you are using a different one
CUDA=cu116
pip install torch==1.13.1+${CUDA} torchvision==0.14.1+${CUDA} torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+${CUDA}.html
pip install torch_geometric==2.2.0
pip install fair-esm==2.0.0
pip install rdkit==2023.9.3
pip install prody==2.4.1
pip install seaborn==0.13.2
pip install tqdm
pip install fairscale
pip install PyYAML
pip install omegaconf
pip install ase
pip install dscribe
pip install MDAnalysis
pip install tensorboardX
pip install meeko
pip install Unidecode
# for python==3.8
pip install gsd==1.9.3
pip install MDAnalysis
pip install treelib
pip install dive-into-graphs
pip install tensorboard
pip install spyrmsd
```

# Download pretrained models

```bash
wget https://zenodo.org/records/11111827/files/data.tar.gz?download=1 -O data.tar.gz
tar xvf data.tar.gz
```

# Run NMDN model
To run prediction, you need the protein structure `PROTEIN.pdb` and docked ligand poses `LIG1.sdf`, `LIG2.sdf`, ...

```bash
python predict.py --prot PROTEIN.pdb --ligs LIG1.sdf LIG2.sdf ...
```

For example:

```bash
python predict.py --prot data/1e66_protein.pdb --ligs data/1e66_1a30/rank1_confidence-0.63.sdf data/1e66_1a30/rank5_confidence-0.94.sdf
```

You will get prediction of the NMDN score and pKd score for each protein-ligand pairs. If you only want to predict the NMDN score, you can add the `--nmdn_only` argument.

```bash
python predict.py --nmdn_only --prot PROTEIN.pdb --ligs LIG1.sdf LIG2.sdf ...
```

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
