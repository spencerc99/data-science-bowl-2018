#!/bin/bash
# Grabs data from kaggle and unzips data
mkdir data
mkdir data/train
mkdir data/test
mkdir data/train/input
mkdir data/train/labels
pip install kaggle
kaggle competitions download -c data-science-bowl-2018
mv stage1_train.zip data/train/input/
unzip data/train/input/stage1_train.zip
mv stage1_train_labels.csv data/train/labels/
mv stage_1_test.zip data/test/
unzip data/test/stage_1_test.zip