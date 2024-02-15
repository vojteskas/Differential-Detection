#!/bin/bash
#PBS -N GMMNCADiff
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l select=1:ncpus=8:mem=100gb:ngpus=1:gpu_mem=15gb:scratch_ssd=100gb
#PBS -l walltime=24:00:00
#PBS -m ae

name="GMMNCADiff"
archivename="$name"_Results.zip

export OMP_NUM_THREADS=$PBS_NUM_PPN

cd "$SCRATCHDIR" || exit 1
mkdir TMPDIR
export TMPDIR=$SCRATCHDIR/TMPDIR
DATADIR=/storage/brno2/home/vojteskas

echo "Creating conda environment"
module add gcc
module add conda-modules-py37
conda create -n DP python=3.10 -y >/dev/null 2>&1
conda activate DP >/dev/null 2>&1

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y >/dev/null 2>&1

echo "Copying project files"
cp $DATADIR/DP/dp.zip .
unzip dp.zip >/dev/null 2>&1

echo "Installing requirements"
pip install -r requirements.txt --cache-dir "$TMPDIR" >/dev/null 2>&1

echo "Copying datasets"
cp -r $DATADIR/deepfakes/datasets/LA.zip .
unzip LA.zip >/dev/null 2>&1

chmod 755 ./*.py

echo "Running the script"
./train_and_eval.py --metacentrum 2>&1

echo "Copying the results"
rm -rf ./*__pycache__*
zip -r $archivename \
    classifiers datasets embeddings feature_processors trainers \
    config.py train_and_eval.py requirements.txt \
    ./*.png ./*.pt \
    >/dev/null 2>&1
cp $archivename $DATADIR/DP/$archivename

clean_scratch
