# How to preprocess single-protein data set to train/evaluate NMDN models

## Step1. Generate Pytorch Geometric (PyG) data set file

In this step, the 3D coordinates from the PDB files are loaded and the pair-wise distance is pre-computed and saved into a PyG file. Check `step1_generate_pyg.py`, you need to modify the several variables.



- `PDB_ROOT`:  First you need to provide PDB files. In DeepAccNet, you untar `/scratch/projects/yzlab/xd638/Protein_MQA/prepared_for_Song/DeepAccNet/DeepAccNet_all_separated_chains_polarH.tar.gz` to somewhere on your disk. Then modify the `PDB_ROOT` variable to the path to the extracted folder.

- `DS_NAME`: Change to your preferred name.

- `CHUNK_ROOT`: This is a temporary folder where some pytorch files are saved. You need to create a folder on Scratch or Vast and change `CHUNK_ROOT` to that folder. Note: This folder is safe to be deleted after `step1_generate_pyg.py` completes.

- `SAVE_ROOT`: This is where you save the result PyG files. Create a folder on Scratch or Vast for that purpose. After running, check if the result file is created at `$SAVE_ROOT/processed/$DS_NAME.pyg`. Check the file size to make sure the file is not empty. The resulting PyG file can be read by Pytorch using `torch.load`.

After setting the variables, do:

```bash
sbatch subjob-step1.sbatch
```

to submit the job.

## Step2. Generate split file

The split file tells the trainning engine how to split the data set into training, evaluation, and test set. You need to set `DS_NAME` and `SAVE_ROOT` in `step2_generate_split.py` and do:

```bash
sbatch subjob-step2.sbatch
```

to submit the job.

## Step3. Run training

In the config file `/scratch/sx801/temp/share/prot-prot/config-exp_pp_014.txt` change the following lines:
```
--data_root=/scratch/sx801/temp/data
--dataset_name=DeepAccNet_test.pyg
--split=DeepAccNet_test.split.pth
```

`--data_root` should be `$SAVE_ROOT`, `--dataset_name` should be `$DS_NAME` and `--split` should be the output split file in step2.