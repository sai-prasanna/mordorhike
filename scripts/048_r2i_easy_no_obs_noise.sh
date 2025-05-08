#!/bin/bash

#SBATCH --array=0-4
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 048_r2i_easy_no_obs_noise
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
seeds=(94 42 1337 13 19)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
job_name=$SLURM_JOB_NAME
logdir="experiments/mordor_hike/${job_name}/${seed}"

echo "Training R2I"
uv run train-r2i --logdir $logdir --configs mordorhike --seed $seed --task "gymnasium_mordor-hike-easy-v0" --batch_size 4 --rssm.deter 1024 --rssm.units 1024 --rssm.hidden 512 --ssm.n_layers 4 --model_opt.lr 0.00017958212993107736 --actor_opt.lr 0.0002341129438718781 --critic_opt.lr 0.0002341129438718781 --.*\\.units 1024 --.*\\.mlp_units 1024 --.*\\.mlp_layers 1 --run.train_ratio 457 --env.kwargs "{'obs_std': 0.0}"

echo "Rolling out R2I"
uv run rollout-r2i --logdir $logdir

echo "Analyzing R2I"
uv run analyze --logdir $logdir
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";

mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out $logdir/
mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err $logdir/