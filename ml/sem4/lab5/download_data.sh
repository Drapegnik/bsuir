#!/usr/bin/env bash

DATA_DIR=data
DATASET=dogs-vs-cats
DATASET_PATH="$DATA_DIR/$DATASET"

mkdir -p "$DATA_DIR"

if [ ! -d "$DATASET_PATH" ]; then
    echo "$DATASET_PATH does not exist, download:"
    # kaggle competitions download -c "$DATASET" -p "$DATA_DIR"
    unzip "$DATASET_PATH.zip" -d "$DATASET_PATH"
    unzip "$DATASET_PATH/train.zip" -d "$DATASET_PATH"
    unzip "$DATASET_PATH/test1.zip" -d "$DATASET_PATH"
else 
    echo "$DATASET_PATH already exist"
fi