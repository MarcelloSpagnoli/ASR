#!/bin/bash
#SBATCH --job-name=tuning_asr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marcello.spagnoli@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=l40
#SBATCH --output=logs/tuning.log
#SBATCH --chdir=/scratch.hpc/marcello.spagnoli2 
#SBATCH --gres=gpu:1

# Comando per l'esecuzione
tirocinio/bin/python3 -u decoder_tuning.py

