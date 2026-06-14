# Dataset Preparation

## PH2

After splitting the PH2 dataset, apply eight-fold augmentation only to the training set.

The augmentation includes the original image, horizontal flipping, and rotations of 90°, 180°, and 270° for both the original and horizontally flipped images.

## ISIC2018

Use `split_isic2018_schemeA.py` to split the ISIC2018 metadata files.
