#!/bin/bash

#SBATCH --array=0-4
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 045_drqn_easy_tuned
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
seeds=(42 1337 13 19 94)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
job_name=$SLURM_JOB_NAME
logdir="experiments/mordor_hike/${job_name}/${seed}"
uv run train-drqn --logdir $logdir --configs defaults --seed $seed --env.name "mordor-hike-easy-v0" --train.batch_size 126 --train.epsilon 0.2389 --drqn.hidden_size 160 --train.learning_rate 0.00022516182798426598 --train.num_gradient_steps 13 --train.train_every 35 --train.target_period 280
uv run rollout-drqn --logdir $logdir
uv run analyze --logdir $logdir

end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";

mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out $logdir/
mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err $logdir/