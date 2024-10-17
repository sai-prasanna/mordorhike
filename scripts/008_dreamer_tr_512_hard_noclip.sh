#!/bin/bash

#SBATCH --array=0-2
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 008_dreamer_tr_512_hard_noclip
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

# Checking the effect of larger train ratio and vector_loss

start=`date +%s`
export OMP_NUM_THREADS=1
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    logdir="experiments/mordor_hike/008_dreamer_tr_512_hard_noclip/42"
    uv run train-dreamer --logdir $logdir --configs mordorhike size2m --run.steps 5e5 --run.save_every 1e5  --run.eval_every 1e5  --seed 42 --run.train_ratio 512  --task mordorhike_hard
    uv run eval-dreamer --logdir $logdir --n_particles 1
    uv run eval-dreamer --logdir $logdir --metric_dir "eval_post_mode" --wandb.group "008_dreamer_tr_512_hard_noclip_post_mode" --wandb.name "42_eval" --policy_mode "eval"  --n_particles 1
    uv run eval-dreamer --logdir $logdir --metric_dir "eval_particle_4" --wandb.group "008_dreamer_tr_512_hard_noclip_eval_particle_4" --wandb.name "42_eval" --n_particles 4
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    logdir="experiments/mordor_hike/008_dreamer_tr_512_hard_noclip/1337"
    uv run train-dreamer --logdir $logdir --configs mordorhike size2m --run.steps 5e5 --run.save_every 1e5  --run.eval_every 1e5  --seed 1337 --run.train_ratio 512  --task mordorhike_hard
    uv run eval-dreamer --logdir $logdir --n_particles 1
    uv run eval-dreamer --logdir $logdir --metric_dir "eval_post_mode" --wandb.group "008_dreamer_tr_512_hard_noclip_post_mode" --wandb.name "1337_eval" --policy_mode "eval" --n_particles 1
    uv run eval-dreamer --logdir $logdir --metric_dir "eval_particle_4" --wandb.group "008_dreamer_tr_512_hard_noclip_eval_particle_4" --wandb.name "1337_eval" --n_particles 4
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    logdir="experiments/mordor_hike/008_dreamer_tr_512_hard_noclip/13"
    uv run train-dreamer --logdir $logdir --configs mordorhike size2m --run.steps 5e5 --run.save_every 1e5  --run.eval_every 1e5  --seed 13 --run.train_ratio 512  --task mordorhike_hard
    uv run eval-dreamer --logdir $logdir --n_particles 1
    uv run eval-dreamer --logdir $logdir --metric_dir "eval_post_mode" --wandb.group "008_dreamer_tr_512_hard_noclip_post_mode" --wandb.name "13_eval" --policy_mode "eval" --n_particles 1
    uv run eval-dreamer --logdir $logdir --metric_dir "eval_particle_4" --wandb.group "008_dreamer_tr_512_hard_noclip_eval_particle_4" --wandb.name "13_eval" --n_particles 4
fi  
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";

mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out $logdir/
mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err $logdir/