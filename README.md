# SSL_transformer

Quick test of SSL transformers for spectrograph (line spectra) data. You could potentially use a ton of different time-series data for pretraining if you are limited by the available spectrograph to train on

Based on the repo https://github.com/microsoft/esvit

[![](imgs/efficient_transf.png)](https://chunyuan.li/assets/pdf/esvit_talk_chunyl.pdf)

![](imgs/esVit.png)
https://paperswithcode.com/sota/self-supervised-image-classification-on
-> https://paperswithcode.com/paper/efficient-self-supervised-vision-transformers

## Getting started with the repo

Downloaded`esvit` as a .zip and added it as a static repo 
(you could have forked or made this as a submodule, now cloned)

Set-up the `venv`:

```
python3.8 -m venv ssl_spectro_venv
source ssl_spectro_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 
```

Manual update now for the PyTorch that you have CUDA 11.x PyTorch 
so you can train/evaluate this on Ampere-generation NVIDIA GPUs (30xx, A100 on AWS, etc)
https://pytorch.org/

```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

[`timm`](https://github.com/rwightman/pytorch-image-models) is "PyTorch image models"

## Getting some sample data

Easiest to get started with the Pytorch Audio data, starting with this https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html

![spectrograph](imgs/voice_spectro.png)

_https://paperswithcode.com/dataset/speech-commands_

**TODO** add some script just to download the data 

### Preprocessing the data

Input audio is as timeseries, we want to make it spectrograph 2D image (in case of audio, this is now time on _x_ and frequency on _y_, i.e. at what times is tour bassline in case with music)

![](imgs/data_in_1D.png)

You could use STFT, Matching Pursuits, Wigner-Ville Transform, Wavelets,  or cEEMD, etc., you have some options

![](imgs/timefreq.jpg)

## Using `EsViT`

See the original [README.md](https://github.com/microsoft/esvit), **you need to download the model (1.2 GB)** if you want to use the pretrained models for finetuning, see [models/README.md](models/README.md)

### Test first the pretrained model on your own data

[![](imgs/transfer_learning.png)](https://chunyuan.li/assets/pdf/esvit_talk_chunyl.pdf)

Replace your project path, other paths are relative to the main path, `=1` with one GPU

```
PROJ_PATH=/home/petteri/PycharmProjects/SSL_spectro
DATA_PATH=$PROJ_PATH/testdata_eval_spectrographs

OUT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300
CKPT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint.pth

cd esvit
python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py --data_path $DATA_PATH --output_dir $OUT_PATH/lincls/epoch0300 --pretrained_weights $CKPT_PATH --checkpoint_key teacher --batch_size_per_gpu 256 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 5 MODEL.NUM_CLASSES 0
```

#### Results of the test

As the data is just random spectrograms with no real labels, you don't expect to learn anything, but you can test for getting the code to run through

```
Max accuracy so far: 28.57%
Epoch: [99]  [0/1]  eta: 0:00:00  lr: 0.000000  loss: 1.459668 (1.459668)  time: 0.730301  data: 0.626781  max mem: 705
Epoch: [99] Total time: 0:00:00 (0.806385 s / it)
Averaged stats: lr: 0.000000  loss: 1.459668 (1.459668)
Test:  [0/1]  eta: 0:00:00  loss: 1.452319 (1.452319)  acc1: 28.571430 (28.571430)  acc5: 100.000008 (100.000008)  time: 0.724810  data: 0.620217  max mem: 705
Test: Total time: 0:00:00 (0.799716 s / it)
* Acc@1 28.571 Acc@5 100.000 loss 1.452
Accuracy at epoch 99 of the network on the 28 test images: 28.6%
Max accuracy so far: 28.57%
Training of the supervised linear classifier on frozen features completed.
Top-1 test accuracy: 28.6
```

### Train with the testing data

See the original [README.md](https://github.com/microsoft/esvit#one-node-training)

[![](imgs/transformer_variants.png)](https://chunyuan.li/assets/pdf/esvit_talk_chunyl.pdf)

[![](imgs/big_CNNs.png)](https://arxiv.org/pdf/2106.09785.pdf)

```
PROJ_PATH=/home/petteri/PycharmProjects/SSL_spectro
DATA_PATH=$PROJ_PATH/testdata_eval_spectrographs

OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/swin_tiny_test_spectro_crap/
python -m torch.distributed.launch --nproc_per_node=1 main_esvit.py --arch swin_tiny --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 2 --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --use_dense_prediction True --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml 
```

![](imgs/garbage_Training.png)

![](imgs/garbage_Training2.png)

Check out the saving frequency also, one model is ~900 MB and now it is saving every 5th epoch for the 300 epoch training (~60 GB)

Depending on your GPU, the batch size has to be quite low for desktop debugging. Batch size of 4 was too much for my `Laptop 3070` with only 6 GB, so with 2 you got this to work. 

If you are using NVIDIA A100, you can hike up the batch size up (40GB or 80GB per GPU awailable)

![](imgs/A100_specs.png)
_https://www.nvidia.com/en-us/data-center/a100/_

#### Check the training log

See [log.txt](utput/esvit_exp/ssl/swin_tiny_test_spectro_crap/log.txt)

**TODO!** Add Tensorboard logging

### Visualizing the network attention

![](imgs/vit_CNN.jpeg)
_[Raghu et al. (2021) "Do Vision Transformers See Like Convolutional Neural Networks?"](https://arxiv.org/abs/2108.08810)_

See original [README.md](https://github.com/microsoft/esvit#analysisvisualization-of-correspondence-and-attention-maps)

Note! Classifier finetuning went to:
`/home/petteri/PycharmProjects/SSL_spectro/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/lincls/epoch0300/checkpoint.pth.tar`

Full training from scratch to:
`/home/petteri/PycharmProjects/SSL_spectro/output/esvit_exp/ssl/swin_tiny_test_spectro_crap/checkpoint.pth`

#### Single Image

For a single image (note that the original README.md was not updated, use `analyze_models.py` instead of `run_analysis.py` that does not exist):

```
PROJ_PATH=/home/petteri/PycharmProjects/SSL_spectro
IMG_PATH=$PROJ_PATH/testdata_eval_spectrographs/train/random_class1/potato_0.png
OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/swin_tiny_test_spectro_crap/explanation/
CKPT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/lincls/epoch0300/checkpoint.pth.tar
SEED=0
python analyze_models.py --arch swin_tiny --image_path $IMG_PATH --output_dir $OUT_PATH --pretrained_weights $CKPT_PATH --learning ssl --seed $SEED --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --vis_attention True --vis_correspondence True MODEL.NUM_CLASSES 0 
```

##### Attention masks

[![](imgs/why_SSL.png)](https://chunyuan.li/assets/pdf/esvit_talk_chunyl.pdf)

Input image for visualization

![](imgs/img_for_attn_viz.png)

Check what these actually are :D

`attn_all_11_query9.png`:

![](imgs/attn_all_11_query9.png)

`attn_masked_all_11_query9.png`:

![](imgs/attn_masked_all_11_query9.png)

`attn_masked_all_11_query9_compressed.png`:

![](imgs/attn_masked_all_11_query9_compressed.png)

##### Correspondence

![](imgs/correspondence0.png)

See the [slides](https://chunyuan.li/assets/pdf/esvit_talk_chunyl.pdf) for quick explanation of this plot

![](imgs/correspondence_slides.png)

For the dataset:

```
PROJ_PATH=/home/petteri/PycharmProjects/SSL_spectro
DATA_PATH=$PROJ_PATH/testdata_eval_spectrographs/
OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/swin_tiny_test_spectro_crap/explanation/
CKPT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/lincls/epoch0300/checkpoint.pth.tar
SEED=0
python analyze_models.py --arch swin_tiny --data_path $DATA_PATH --output_dir $OUT_PATH --pretrained_weights $CKPT_PATH --learning ssl --seed $SEED --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml  --measure_correspondence True MODEL.NUM_CLASSES 0 
```

Saves `measure_correspondence.pickle`

## What next?

* **Get more training data** Do a proper search for available datasets, that could be similar to the spectrographs


* **Explore time-frequency transforms** You could do multiple transformations. [STFT](https://pytorch.org/audio/stable/transforms.html) the fastest, but not very accurate; [Wavelets, PyCWT](https://pycwt.readthedocs.io/en/latest/), [Matching Pursuit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html), [Wigner-Ville distribution, e.g. from the Time-Frequency Toolbox](https://tftb.readthedocs.io/en/latest/auto_examples/plot_4_1_3_chirps_wvd.html), [EEMD](https://bitbucket.org/luukko/libeemd/src/master/), etc. You could have multiple transforms all at once as data augmentations, or just pick one. Quantitative method for quantifying the "spectral spectrograph"-likeness of our time-frequency 2D images. 


* **Would there be more light-weight** transformers, hybrid CNN-transformers, that would train even faster with good performance?

### Dataset seeds

Get time-frequency plots from any time series (audio, music, voice, EEG, ECG, any biosignals, stock markets), and get line spectra from hyperspectral cubes.

#### Line Spectra

* ?

#### Hyperspectral Cubes

You could just take line spectra from cubes (columns and rows)

* **DiaRetDB2 spectral fundus image database (see [Fält et al. 2011](https://doi.org/10.2352/J.ImagingSci.Technol.2011.55.3.030509) and [Laaksonen 2016](https://lutpub.lut.fi/bitstream/handle/10024/123509/Lauri%20Laaksonen%20A4.pdf?sequence=2)):** Spectral channels were separated using 30 narrow bandpass interference filters in the visual range from 400 to 700 nm. A monochrome digital charge-coupled device camera was used as a detector. Using this system, spectral fundus images were recorded from 72 voluntary human subjects: 55 diabetic patients and 17 healthy control subjects.

* [Torbjørn Skaulia and Joyce Farrell (2013)](https://web.stanford.edu/~jefarrel/Publications/2010s/2013_HyperspectralImagingDatabase.pdf): "A collection of hyperspectral images for imaging systems research" - [Cited by 59](https://scholar.google.co.uk/scholar?hl=en&as_sdt=0%2C5&q=A+collection+of+hyperspectral+images+for+imaging+systems+research+&btnG=)   

#### Time Series

Transformers also for sensor (IoT) data, [Zhang et al 2021: Soft Sensing Transformer: Hundreds of Sensors are Worth a Single Word](https://arxiv.org/pdf/2111.05973.pdf)

##### Audio / Voice

A lot of audio/voice available readily curated. But if you had like a startup doing voice-based analysis for AD, you could scrape the whole internet for whatever Yotuube video clips, and all the songs that you can?

* https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad
* https://towardsdatascience.com/40-open-source-audio-datasets-for-ml-59dc39d48f06
* https://www.twine.net/blog/100-audio-and-video-datasets/
* https://github.com/jim-schwoebel/voice_datasets

##### Biosignals

* **[PhysioNet](https://physionet.org/about/database/)**
* [Wagner et al. 2021 "PTB-XL, a large publicly available electrocardiography dataset"](https://doi.org/10.1038/s41597-020-0495-6) ECG
* [Zhang et al. 2021: "Eegdenoisenet: A benchmark dataset for deep learning solutions of eeg denoising"](https://github.com/ncclabsustech/EEGdenoiseNet)

##### Stock Market

* [10 Best Stock Market Datasets for Machine Learning ](https://hackernoon.com/10-best-stock-market-datasets-for-machine-learning-e71437qm)

## Low-level fixes needed

* Train directly using **monochrome images** instead of making spectrographs RGB to match the Github code from Microsoft
 

* Do the preprocessing better so you can get high-res spectrographs


* Explore maybe some [CycleGANs](https://github.com/simontomaskarlsson/GAN-MRI) for doing the transformation from spectrograph-like to spectrograph images? Or do we just overfit again the generator?

![](imgs/CycleGAN_T2_hist.gif)
_CycleGAN - [T1](https://radiopaedia.org/articles/t1-weighted-image) to [T2](https://radiopaedia.org/articles/t2-weighted-image?lang=gb). Left: Input image. Middle: Synthetic images generated during training. Right: Ground truth. - https://github.com/simontomaskarlsson/GAN-MRI_


* Try to actually understand the hyperparameters as well. Does the training really converge easily if you had signal in your dataset? Do we want to do some hyperparameter tuning? Or test this first with some smallish dataset (MNIST, or are there some small voice/audio datasets that are commonly used?) so that you get a sense on how the network behaves?
