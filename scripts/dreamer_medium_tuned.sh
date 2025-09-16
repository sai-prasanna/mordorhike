#!/bin/bash

#SBATCH --array=0-4
#SBATCH --job-name dreamer_medium_tuned
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

start=`date +%s`
export OMP_NUM_THREADS=1
seeds=(42 1337 13 19 94)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
job_name=$SLURM_JOB_NAME
logdir="experiments/mordor_hike/${job_name}/${seed}"
uv run train-dreamer --logdir $logdir --configs mordorhike --seed $seed --task "gymnasium_mordor-hike-medium-v0" --batch_size 8 --dyn.rssm.deter 256 --dyn.rssm.hidden 85 --dyn.rssm.classes 11 --opt.lr 0.00023854151368141174 --run.train_ratio 262 --.*\\.units 165
uv run rollout-dreamer --logdir $logdir
uv run analyze --logdir $logdir
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";