#!/bin/bash
#PBS -N DP_FFLSTM2_DF21nonVC
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=200gb:ngpus=1:gpu_mem=20gb:scratch_ssd=100gb
#PBS -m ae

export OMP_NUM_THREADS=$PBS_NUM_PPN


name=DP_FFLSTM2_DF21nonVC
archivename="$name"_Results.zip
DATADIR=/storage/brno2/home/vojteskas


cd "$SCRATCHDIR" || exit 1
mkdir TMPDIR
export TMPDIR="$SCRATCHDIR/TMPDIR"


echo "Creating conda environment"
module add gcc
module add conda-modules-py37
conda create --prefix "$TMPDIR/condaenv" python=3.10 -y >/dev/null 2>&1
conda activate "$TMPDIR/condaenv" >/dev/null 2>&1
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y >/dev/null 2>&1


echo "Copying project files"
cp $DATADIR/DP/dp.zip .
unzip dp.zip >/dev/null 2>&1


echo "Installing project requirements"
pip install -r requirements.txt --cache-dir "$TMPDIR" >/dev/null 2>&1


echo "Copying dataset(s)"
cp -r $DATADIR/deepfakes/datasets/DF21.tar.gz .
tar -xvzf DF21.tar.gz >/dev/null 2>&1


cp -r $DATADIR/deepfakes/datasets/LA19.tar.gz .
tar -xvzf LA19.tar.gz >/dev/null 2>&1


chmod 755 ./*.py
echo "Running the script"
./train_and_eval.py --metacentrum --dataset ASVspoof2021DFDataset_nonVC_pair --extractor XLSR_300M --processor MHFA --classifier FFLSTM2 --num_epochs 20 2>&1


echo "Copying results"
find . -type d -name "__pycache__" -exec rm -rf {} +
zip -r "$archivename" classifiers datasets embeddings feature_processors trainers ./*.py ./*.png ./*.pt >/dev/null 2>&1
cp "$archivename" $DATADIR/DP/$archivename >/dev/null 2>&1


clean_scratch