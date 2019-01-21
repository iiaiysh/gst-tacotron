import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import audio

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  if args.mel_targets is not None:
    mel_targets = np.load(args.mel_targets)
  else:
    mel_targets = None

  if args.reference_audio is not None:
    ref_wav = audio.load_wav(args.reference_audio)
    reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
  else:
    reference_mel = None
  
  synth = Synthesizer(mel_targets=mel_targets, reference_mel=reference_mel, reuse=reuse)
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)


  with open(path, 'wb') as f:
    print('Synthesizing: %s' % args.text)
    print('Output wav file: %s' % path)
    f.write(synth.synthesize(args.text)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--text', required=True, default=None, help='Single test text sentence')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--reference_audio', default=None, help='Reference audio path')
  parser.add_argument('--mel_targets', default=None, help='Mel-targets path, used when use teacher_force generation')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
