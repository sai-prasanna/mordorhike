#!/bin/bash

#SBATCH --array=0-50%10
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 039_tune_drqn_medium_new
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
job_name=$SLURM_JOB_NAME
uv run tune-drqn --env.name "mordor-hike-medium-v0" --neps_root_directory experiments/neps_drqn_medium_new
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";
