# SSL_transformer

Based on the repo https://github.com/microsoft/esvit

![](imgs/esVit.png)
https://paperswithcode.com/sota/self-supervised-image-classification-on
-> https://paperswithcode.com/paper/efficient-self-supervised-vision-transformers

## Getting started with the repo

Clone the `esvit` (you could have forked or made this as a submodule, now cloned)

```
git clone https://github.com/microsoft/esvit
```

Set-up the `venv`:

```
python3.8 -m venv ssl_spectro_venv
source ssl_spectro_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 
```

## Getting some sample data

Easiest to get started with the Pytorch AUdio data, starting with this https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html

![spectrograph](imgs/voice_spectro.png)

_https://paperswithcode.com/dataset/speech-commands_

**TODO** add some script just to download the data 

### Preprocessing the data

Input audio is as timeseries, we want to make it spectrograph 2D image (in case of audio, this is now time on _x_ and frequency on _y_, i.e. at what times is tour bassline in case with music)

You could use STFT, Matching Pursuits, Wigner-Ville Transform, Wavelets,  or cEEMD, etc., you have some options

![](imgs/timefreq.jpg)