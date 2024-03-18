# Running NDMN model for protein-protein interaction prediction

## Cloning the repo

```bash
git clone https://github.com/SongXia-NYU/PhysDime_dev.git
cd ./PhysDime_dev
```

Unless specified otherwise, all scripts should run in the `PhysDime_dev` directory.

## Training

To run NMDN on DockGround data set, you first obtain the config file by:

```bash
cp /scratch/sx801/temp/share/prot-prot/config-exp_pp_007.txt .
```

To initialized a trainning job, you need to:

```bash
source /vast/sx801/venv/bin/activate
python smart_job_submit.py config-exp_pp_007.txt --debug
```

You will get a sbatch file `SUBMIT_TRAIN-exp_pp_007.sbatch`. Open it, it should be straightforward.

To actually submit the job:

```bash
python smart_job_submit.py config-exp_pp_007.txt
```

After the job runs, you will get a folder `exp_pp_007_run_xxxx-xx-xx_xxxxxx__xxxxxx` which contains all information about this run.

## Testing

Once the folder `exp_pp_007_run_xxxx-xx-xx_xxxxxx__xxxxxx` is generated, you can schedule a testing job by:

```bash
python smart_job_submit.py exp_pp_007_run_xxxx-xx-xx_xxxxxx__xxxxxx
```

The job will NOT run immediately, instead, it be automatically scheduled after the completion of the trainning job.

To check scheduled jobs, run:

```bash
squeue -u $USER
```
