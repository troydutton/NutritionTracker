#!/bin/bash

cd data/OID

# Create the new directory structure
mkdir -p train/images train/labels valid/images valid/labels

# Move train images and labels to the new structure
mv images/train/* train/images/
mv labels/train_coco/* train/labels/

# Move validation images and labels to the new structure
mv images/valid/* valid/images/
mv labels/val_coco/* valid/labels/

# Optionally, remove old directories if they are now empty
rmdir images/train images/valid labels/train_coco labels/val_coco
rmdir images labels
