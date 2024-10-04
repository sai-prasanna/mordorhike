#!/bin/bash

#SBATCH --array=0-2
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 001_dreamer_tr_128
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

# Checking the effect of smaller train ratio and vector_loss

start=`date +%s`
export OMP_NUM_THREADS=1
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    logdir="experiments/mordor_hike/001_dreamer_tr_128/42"
    uv run train-dreamer --logdir $logdir --configs mordorhike size2m --run.steps 5e5 --run.save_every 1e5  --run.eval_every 1e5 --seed 42 --run.train_ratio 128
    uv run eval-dreamer --logdir $logdir
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    logdir="experiments/mordor_hike/001_dreamer_tr_128/1337"
    uv run train-dreamer --logdir $logdir --configs mordorhike size2m --run.steps 5e5 --run.save_every 1e5  --run.eval_every 1e5 --seed 1337 --run.train_ratio 128
    uv run eval-dreamer --logdir $logdir
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    logdir="experiments/mordor_hike/001_dreamer_tr_128/13"
    uv run train-dreamer --logdir $logdir --configs mordorhike size2m --run.steps 5e5 --run.save_every 1e5  --run.eval_every 1e5 --seed 13 --run.train_ratio 128
    uv run eval-dreamer --logdir $logdir
fi  
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";

mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out $logdir/
mv experiments/slurm/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err $logdir/