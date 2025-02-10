#!/bin/bash

#SBATCH --array=0-2
#SBATCH --partition mlhiwidlc_gpu-rtx2080
#SBATCH --job-name 040_drqn_medium_tuned_510
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

 #{'batch_size': 33, 'env_steps': 300000, 'epsilon': 0.21715511625469264, 'hidden_size': 434, 'learning_rate': 0.0005326262097599815, 'num_gradient_steps': 15, 'num_layers': 2, 'train_every': 51}
start=`date +%s`
export OMP_NUM_THREADS=1
seeds=(42 1337 13)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
job_name=$SLURM_JOB_NAME
logdir="experiments/mordor_hike/${job_name}/${seed}"
uv run train-drqn --logdir $logdir --configs defaults --seed $seed --env.name "mordor-hike-medium-v0" --train.learning_rate 0.0005326262097599815 --train.batch_size 33  --train.epsilon 0.21715511625469264 --drqn.hidden_size 434 --drqn.num_layers 2 --train.num_gradient_steps 15 --train.train_every 51 --train.target_period 510
uv run rollout-drqn --logdir $logdir
uv run analyze --logdir $logdir
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";

mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out $logdir/
mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err $logdir/