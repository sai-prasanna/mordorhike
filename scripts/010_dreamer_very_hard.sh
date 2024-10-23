#!/bin/bash

#SBATCH --array=0-2
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 010_dreamer_very_hard
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

# Check the performance of the very hard task. Ideally we want the performance to drop.

start=`date +%s`
export OMP_NUM_THREADS=1
seeds=(42 1337 13)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
job_name=$SLURM_JOB_NAME
logdir="experiments/mordor_hike/${job_name}/${seed}"
# uv run train-dreamer --logdir $logdir --configs mordorhike size2m --run.steps 5e5 --run.save_every 1e5  --run.eval_every 1e5  --seed $seed --run.train_ratio 512  --task mordorhike_veryhard --n_particles 1
# uv run eval-dreamer --logdir $logdir --mine_deep_set_size 256
# uv run eval-dreamer --logdir $logdir --metric_dir "eval_particle_4" --wandb.group "${job_name}_eval_particle_4" --wandb.name "${seed}_eval" --n_particles 4 --mine_deep_set_size 256
uv run eval-dreamer --logdir $logdir --metric_dir "eval_particle_10" --wandb.group "${job_name}_eval_particle_10" --wandb.name "${seed}_eval" --n_particles 10 --mine_deep_set_size 64

end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";

mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out $logdir/
mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err $logdir/