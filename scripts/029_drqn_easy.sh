#!/bin/bash

#SBATCH --array=0-2
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 029_drqn_easy
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";


start=`date +%s`
export OMP_NUM_THREADS=1
seeds=(42 1337 13)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
job_name=$SLURM_JOB_NAME
logdir="experiments/mordor_hike/${job_name}/${seed}"
uv run train-drqn --logdir $logdir --configs defaults --seed $seed --env.name "mordor-hike-easy-v0"
uv run rollout-drqn --logdir $logdir
uv run analyze --logdir $logdir
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";

mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out $logdir/
mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err $logdir/