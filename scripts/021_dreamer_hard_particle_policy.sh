#!/bin/bash

#SBATCH --array=0-2
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 021_dreamer_hard_particle_policy
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

# Runing experiment with particle policy where actor-critic is still trained on one particle in imagination.
start=`date +%s`
export OMP_NUM_THREADS=1
seeds=(42 1337 13)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
job_name=$SLURM_JOB_NAME
logdir="experiments/mordor_hike/${job_name}/${seed}"
uv run train-dreamer --logdir $logdir --configs mordorhike --task gymnasium_mordor-hike-hard-v0 --seed $seed --n_particles 10 --particle_policy True
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";

mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out $logdir/
mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err $logdir/