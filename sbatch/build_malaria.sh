#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cecile_meier-scherling@brown.edu
#SBATCH --output=sbatch/logs-sim/sim_%A.out
#SBATCH --error=sbatch/logs-sim/sim_%A.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH -N 1


uv sync --extra cuda12
uv run --extra cuda12 python3 -u -m examples.build_malaria_dataset