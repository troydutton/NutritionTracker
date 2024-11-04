#!/bin/bash

mkdir data

curl -L -o data/archive.zip https://www.kaggle.com/api/v1/datasets/download/sainikhileshreddy/food-recognition-2022

unzip data/archive.zip -d data

rm data/archive.zip

rm -r data/hub