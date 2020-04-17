#!/usr/bin/env bash

DATA_DIR=data
AUTHOR=rtatman
DATASET_NAME=glove-global-vectors-for-word-representation
DATASET_PATH="$DATA_DIR/$DATASET_NAME"

mkdir -p "$DATA_DIR"

if [ ! -d "$DATASET_PATH" ]; then
    echo "$DATASET_PATH does not exist, download:"
    kaggle datasets download "$AUTHOR/$DATASET_NAME" -p "$DATA_DIR"
    unzip "$DATASET_PATH.zip" -d "$DATASET_PATH"
else 
    echo "$DATASET_PATH already exist"
fi