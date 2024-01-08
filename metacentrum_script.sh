#!/bin/bash
#PBS -N SSL_Spoofing
#PBS -q phi@cerit-pbs.cerit-sc.cz
#PBS -l select=1:ncpus=256:mem=385gb:scratch_ssd=100gb
#PBS -l walltime=96:00:00
#PBS -m ae

export OMP_NUM_THREADS=$PBS_NUM_PPN

cd "$SCRATCHDIR" || exit 1
mkdir TMPDIR
export TMPDIR=$SCRATCHDIR/TMPDIR
DATADIR=/storage/brno2/home/vojteskas

module add gcc
module add conda-modules-py37
conda create -n SSL_Spoofing python=3.7 -y
conda activate SSL_Spoofing

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html --cache-dir "$TMPDIR"

git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
cd SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 || exit 2
pip install --editable ./ --cache-dir "$TMPDIR"
cd ..
pip install -r requirements.txt --cache-dir "$TMPDIR"
cd ..

cp $DATADIR/DP/dp.zip .
unzip dp.zip

pip install -r requirements.txt --cache-dir "$TMPDIR"

cp -r $DATADIR/deepfakes/datasets/LA.zip .
unzip LA.zip >/dev/null 2>&1
cp $DATADIR/DP/xlsr2_300m.pt .

chmod 755 ./*.py

./train.py
./eval.py

cp TrainingLossAndAccuracy.png $DATADIR/DP/TrainingLossAndAccuracy.png
cp diffmodel.pt $DATADIR/DP/diffmodel.pt

clean_scratch
