import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import blizzard, ljspeech, blizzard2013
from hparams import hparams


def preprocess_blizzard(args):
  in_dir = os.path.join(args.base_dir, 'Blizzard2012')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = blizzard.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)

def preprocess_ysh(args):
  #in_dir = os.path.join(args.base_dir, 'database/LJSpeech-1.0')
  #in_dir = os.path.join(args.base_dir, 'database/LJSpeech-1.1')
  in_dir = os.path.join(args.base_dir, args.dataset_path)
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata, csv_name_list = ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir, csv_name_list)

def preprocess_ljspeech(args):
  #in_dir = os.path.join(args.base_dir, 'database/LJSpeech-1.0')
  #in_dir = os.path.join(args.base_dir, 'database/LJSpeech-1.1')
  in_dir = os.path.join(args.base_dir, 'database/audiobook_manual')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata, csv_name_list = ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir, csv_name_list)

def preprocess_blizzard2013(args):
  in_dir = os.path.join(args.base_dir, 'database/blizzard2013/segmented')
  out_dir = os.path.join(args.base_dir, args.output)
  os.makedirs(out_dir, exist_ok=True)
  metadata = blizzard2013.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
  write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir, csv_name_list):
  with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    for i,m in enumerate(metadata):
      f.write('|'.join([str(x) for x in m]) + '|' + csv_name_list[i]  + '\n')
  frames = sum([m[2] for m in metadata])
  hours = frames * hparams.frame_shift_ms / (3600 * 1000)
  print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
  print('Max input length:  %d' % max(len(m[3]) for m in metadata))
  print('Max output length: %d' % max(m[2] for m in metadata))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.getcwd())
  parser.add_argument('--output', default='training')
  parser.add_argument('--dataset', default='ysh', choices=['ysh','blizzard', 'ljspeech', 'blizzard2013'])
  parser.add_argument('--num_workers', type=int, default=cpu_count())
  parser.add_argument('--dataset_path', required=True)
  args = parser.parse_args()
  if args.dataset == 'blizzard':
    preprocess_blizzard(args)
  elif args.dataset == 'ljspeech':
    preprocess_ljspeech(args)
  elif args.dataset == 'blizzard2013':
    preprocess_blizzard2013(args)
  elif args.dataset == 'ysh':
    preprocess_ysh(args)


if __name__ == "__main__":
  main()
