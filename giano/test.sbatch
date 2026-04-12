#!/bin/bash
#SBATCH --job-name=asr_train_l40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marcello.spagnoli@studio.unibo.it
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=l40
#SBATCH --output=logs/test.log
#SBATCH --chdir=/scratch.hpc/marcello.spagnoli2 
#SBATCH --gres=gpu:1

# Comando per l'esecuzione
tirocinio/bin/python3 -u test.py








