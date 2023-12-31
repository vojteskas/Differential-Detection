#!/bin/bash
#PBS -N SSL_Spoofing
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l select=1:ncpus=2:mem=64gb:ngpus=1:scratch_ssd=100gb
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -m ae

### !!! CHANGE THE ROOT PATH !!! ###

cd $SCRATCHDIR
mkdir TMPDIR
export TMPDIR=$SCRATCHDIR/TMPDIR
DATADIR=/storage/brno2/home/vojteskas

module add gcc
module add conda-modules-py37
conda create -n SSL_Spoofing python=3.7 -y
conda activate SSL_Spoofing

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html --cache-dir $TMPDIR

git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
cd SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./ --cache-dir $TMPDIR
cd ..
pip install -r requirements.txt --cache-dir $TMPDIR
cd ..

cp $DATADIR/DP/dp.zip .
unzip dp.zip

pip install -r requirements.txt --cache-dir $TMPDIR

cp -r $DATADIR/deepfakes/datasets/LA.zip .
unzip LA.zip 2>1 > /dev/null
cp $DATADIR/DP/xlsr2_300m.pt .

chmod 755 *.py

./train.py
./eval.py

cp TrainingLossAndAccuracy.png $DATADIR/DP/TrainingLossAndAccuracy.png
cp diffmodel.pt $DATADIR/DP/diffmodel.pt

clean_scratch
