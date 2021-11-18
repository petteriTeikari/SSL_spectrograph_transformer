##

# PyCharm Cell mode
# https://plugins.jetbrains.com/plugin/7858-pycharm-cell-mode/versions

##

# from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm.notebook import tqdm
from loguru import logger

##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(" device (for training) = '{}'", device, feature="f-strings", colorize=True)

##

from torchaudio.datasets import SPEECHCOMMANDS
import os

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        # Dowwnloads to your repo path now, e.g.
        # /home/petteri/PycharmProjects/SSL_spectro/SpeechCommands
        super().__init__("./", download=True)
        # download_path = '/home/petteri/voicecommands_data'
        # if not os.path.exists(download_path):
        #     os.mkdir(download_path)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

# A data point in the SPEECHCOMMANDS dataset is a tuple made of a waveform (the audio signal),
# the sample rate, the utterance (label), the ID of the speaker, the number of the utterance.

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.plot(waveform.t().numpy())
plt.show()

##

# here, go through all the .WAVs and convert to spectrographs
# e.g. TorchAudio, https://pytorch.org/audio/stable/transforms.html