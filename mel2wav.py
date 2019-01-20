import numpy as np
from util import audio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mel',requried=True)
parser.add_argument('--wav',requried=True)
args = parser.parse_args()


linear_target = np.load(args.mel)

waveform_target = audio.inv_spectrogram(linear_target.T)
audio.save_wav(waveform_target, args.wav)
