# UniEMIR

## UniEMIR WebUI

We developed a web-based user interface, which can be deployed on high-performance GPU servers. You can use the colab <a target="_blank" href="https://colab.research.google.com/github/ryanhe312/UniEMIR/blob/main/UniEMIR.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> to run the Web UI or employ the following steps to run the web user interface locally.

1. Install Packages

```
conda create -n uniemir python=3.11.8
conda activate uniemir
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

3. Run the Web User Interface

```
python app.py
```

Then, you can visit the web interface at [http://127.0.0.1:7860/](http://127.0.0.1:7860/). We provide a video tutorial for UniEMIR web user interface.

[![](https://markdown-videos-api.jorgenkh.no/youtube/psoT_a0Jg3U)](https://youtu.be/psoT_a0Jg3U)

## UniEMIR Plugin

UniEMIR plugin is an out-of-the-box extensions for the [ImageJ](https://imagej.net/ij/), allowing for the restoration of 2D or 3D electron microscopy images on personal computers with CPU or GPU processing capabilities. You can download UniEMIR ImageJ plugin at [https://github.com/ryanhe312/UniEMIR/releases](https://github.com/ryanhe312/UniEMIR/releases) and unzip to the ImageJ folder (the one with ImageJ-win64.exe). Please also make sure that you have [Visual Studio](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) installed.

Users can simply open an image and run the plugin to obtain results. The plugin accepts 2 (xy) and 3 (zxy) dimensional images in uint8 data type. We provide a video tutorial for UniEMIR plugin.

[![](https://markdown-videos-api.jorgenkh.no/youtube/fYaCG96sAcw)](https://youtu.be/fYaCG96sAcw)

## Training and Testing

### 1. Prepare Datasets

All training and testing data involved in the experiments come from existing literatures. You can download our preprocessed data from the zenodo [repository](https://doi.org/10.5281/zenodo.12738837) and unzip them into the corresponding folders. Or you can prepare the dataset according to the [instruction](data/dataset.md). Then, please edit the dataset path in `config/*.json`.

### 2. Inferencing Model

You can run the following command to test UniEMIR performance on given dataset.

* Denoising:

```bash
python run.py  -p test -c config/UniEMIR-denoise.json -b 16 --gpu 0 --resume experiments/train_UniEMIR-denoise/checkpoint/300
```

* Super-resolution

```bash
python run.py  -p test -c config/UniEMIR-zoom.json -b 16 --gpu 0 --resume experiments/train_UniEMIR-zoom/checkpoint/300
```

* Isotropic Reconstruction

```bash
python run.py  -p test -c config/UniEMIR-isotropic.json -b 16 --gpu 0 --resume experiments/train_UniEMIR-isotropic/checkpoint/300
```

The metrics for each test image can further be calculated by calling `python metric.py path/to/output/dir`. For example, `python metric.py experiments/test_UniEMIR-denoise_240713_172734`.

### 3. Training Model

You can run the following command to pretrain and finetune UniEMIR models.

1. Pretraining model on multiple tasks:

```bash
python run.py -c config/UniEMIR.json -b 6 --gpu 0
```

2. Finetuning model on single task:

```bash
python run.py -c config/UniEMIR-denoise.json -b 6 --gpu 0 --resume experiments/train_UniEMIR/checkpoint/200
python run.py -c config/UniEMIR-zoom.json -b 6 --gpu 0 --resume experiments/train_UniEMIR/checkpoint/200
python run.py -c config/UniEMIR-isotropic.json -b 6 --gpu 0 --resume experiments/train_UniEMIR/checkpoint/200
```

### 4. Exporting model for ImageJ plugin 

You can run the following command to build models for ImageJ plugin. Arguments can be `super-resolution`, `denoising` and `isotropic_reconstruction`.

```bash
python export.py super-resolution
```

And you can unzip the models under `models` folder in ImageJ root folder.

## CITE

```bibtex
@software{he2024pushing,
  author = {Ruian He, Weimin Tan, Chenxi Ma and Bo Yan},
  doi = {10.5281/zenodo.12738837},
  title = {UniEMIR},
  url = {https://github.com/ryanhe312/UniEMIR},
  year = {2024}
}
```
