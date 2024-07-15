# Preparing Datasets

### 1. Download the datasets

You can download the datasets from the following links.

* EMDiffuse: https://zenodo.org/record/8136295
* OpenOrganelle: https://openorganelle.janelia.org/datasets
* FANC: https://bossdb.org/project/phelps_hildebrand_graham2021
* MICrONS: https://bossdb.org/project/microns-minnie

For `EMDiffuse`, just download the whole dataset as a zip file. For `OpenOrganelle`, `FANC` and `MICrONS`, we provide a script for downloading with Python at `data/download_dataset.py`.

### 2. Reorganize the datasets

Please reorganize the datasets as the following strcture:

```
EMDiffuse/EMDiffuse_dataset
    BM/
    Heart/
    Liver/
    HeLa/
    brain_train/
    brain_test/

FANC/FANC/
    em_tif/
    em_tif_test/

MICrONS/minnie65_8x8x40/
    em_tif/
    em_tif_test/

OpenOrganelle/
    jrc_mus-kidney.n5/
        em/fibsem-uint8/
            s0_tif/
            s0_tif_test/
    jrc_mus-liver.n5/
        em/fibsem-uint8/
            s0_tif/
            s0_tif_test/
```

### 3. Split train and test sets

For the BM, Heart, and Liver datasets, we split the training dataset and the test dataset in the ratio of 80/20. That is, we use the first 8 slices of the BM and Heart datasets as the training set and the last 2 as the test set. For the Liver dataset, we use the first 16 slices as the training set and the last 4 as the test set. And we reorganize the folders as follows.

```
EMDiffuse/EMDiffuse_dataset
    BM_train/
        1/
        ...
        8/
    BM_test/
        9/
        10/
    Heart_train/
        1/
        ...
        8/
    Heart_test/
        9/
        10/
    Liver_train/
        1/
        ...
        16/
    Liver_test/
        17/
        ...
        20/
```


### 4. Preprocessing datasets

For SR datasets (brain_train/brain_test), please run the following commands to prepare the aligned dataset.

```bash
cd data/RAFT/core
python super_res_register.py --path path/to/dataset --patch_size 128 --overlap 0.125
```

For denoising datasets (brain_train/brain_test/BM/Liver/Heart/HeLa), please run the following commands to prepare the aligned dataset. Please replace the `Brain` to the dataset name if you use for other datasets.

```bash
cd data/RAFT/core
python register.py --path path/to/dataset --tissue Brain --patch_size 256 --overlap 0.125
```

For anisotropic datasets (MICrONS/FANC), please run the following command to transpose the volume, and use transposed volume for training.

```bash
python data/transpose.py path/to/dataset
```

### (Optional) Reconstructing Isotropic Volumes

After the isotropic reconstruction, you may want to reconstruct the whole volume for visualization. You can use the following command to reconstruct the volume from the cropped images.

```bash
python data/reconstruct.py path/to/output/dir/with/tif/files
```