#!/bin/bash

#SBATCH --array=0-2
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name powm
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";


start=`date +%s`

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    uv run train-dreamer --logdir experiments/mordor_hike/model_2m_level_medium_tr_512/42 --configs mordorhike size2m --run.steps 2e5 --run.save_every 2e4  --run.eval_every 2e4 --run.num_envs 16 --seed 42
    uv run eval-dreamer --logdir experiments/mordor_hike/model_2m_level_medium_tr_512/42
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    uv run train-dreamer --logdir experiments/mordor_hike/model_2m_level_medium_tr_512/13 --configs mordorhike size2m --run.steps 2e5 --run.save_every 2e4  --run.eval_every 2e4 --run.num_envs 16 --seed 13
    uv run eval-dreamer --logdir experiments/mordor_hike/model_2m_level_medium_tr_512/13
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    uv run train-dreamer --logdir experiments/mordor_hike/model_2m_level_medium_tr_512/1337 --configs mordorhike size2m --run.steps 2e5 --run.save_every 2e4  --run.eval_every 2e4 --run.num_envs 16 --seed 1337
    uv run eval-dreamer --logdir experiments/mordor_hike/model_2m_level_medium_tr_512/1337
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    echo "blah"
fi

end=`date +%s`

echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";
