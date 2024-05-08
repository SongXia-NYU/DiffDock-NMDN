CASF-2016-scoring-ranking: Blinding docking on the 285 protein-ligand pairs for scoring and ranking power evaluation:
```
CASF-2016-scoring-ranking
└── ${PDB}_diffdock_nmdn.sdf
```

CASF-2016-screening: cross docked poses for CASF-2016 screening power evaluation

```
CASF-2016-screening
└── ${TARGET_PDB}
    └── ${TARGET_PDB}_${LIGAND_PDB}_diffdock_nmdn.sdf
```

MerckFEP:

```
MerckFEP
└── ${TARGET_NAME}.${LIGAND_NAME}_diffdock_nmdn.sdf
```

LIT-PCBA:

```
LIT-PCBA
└── ${TARGET_NAME}
    └── ${LIGAND_ID}_${TARGET_PDB}_diffdock_nmdn.sdf
```

trained_model: trained NMDN model. Used in https://github.com/SongXia-NYU/DiffDock-NMDN.