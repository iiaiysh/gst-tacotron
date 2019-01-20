import numpy as np
from util import audio
linear_target = np.load(args.mel)

waveform_target = audio.inv_spectrogram(linear_target.T)
audio.save_wav(waveform_target, args.wav)