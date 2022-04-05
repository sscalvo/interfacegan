# python3.7
"""Generates a collection of images with specified model.

Commonly, this file is used for data preparation. More specifically, before
exploring the hidden semantics from the latent space, user need to prepare a
collection of images. These images can be used for further attribute prediction.
In this way, it is able to build a relationship between input latent codes and
the corresponding attribute scores.
"""

import os.path
from os import mkdir
import argparse
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Generate images with given model.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-i', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images to generate. This field will be '
                           'ignored if `latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('-S', '--generate_style', action='store_true',
                      help='If specified, will generate layer-wise style codes '
                           'in Style GAN. (default: do not generate styles)')
  parser.add_argument('-I', '--generate_image', action='store_false',
                      help='If specified, will skip generating images in '
                           'Style GAN. (default: generate images)')
  parser.add_argument('-c', '--file_name_counter', type=int, default=0,
                      help='Number upon which image file names will be generated'
                      ' for being saved (required)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()

  logger = setup_logger(args.output_dir, logger_name='generate_data')

  logger.info(f'Initializing generator.')
  gan_type = MODEL_POOL[args.model_name]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(args.model_name, logger)
    kwargs = {}
  elif gan_type == 'stylegan':
    model = StyleGANGenerator(args.model_name, logger)
    kwargs = {'latent_space_type': args.latent_space_type}
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')
  os.mkdir(f'{args.output_dir}/images')
  os.mkdir(f'{args.output_dir}/w')

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.latent_codes_path):
    logger.info(f'  Load latent codes from `{args.latent_codes_path}`.')
    latent_codes = np.load(args.latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
  else:
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.easy_sample(args.num, **kwargs)
  total_num = latent_codes.shape[0]

  logger.info(f'Generating {total_num} samples.')
  results = defaultdict(list)
  fnc = args.file_name_counter # sscalvo
  for latent_codes_batch in model.get_batch_inputs(latent_codes):
    if gan_type == 'pggan':
      outputs = model.easy_synthesize(latent_codes_batch)
    elif gan_type == 'stylegan':
      outputs = model.easy_synthesize(latent_codes_batch,
                                      **kwargs,
                                      generate_style=args.generate_style,
                                      generate_image=args.generate_image)

    for w,image in zip(outputs['w'], outputs['image']): # sscalvo modified
      # Taking advantage of StyleGAN2 face alignement, let's crop both eyes 
      # and save the cropped image to a 224x224 file
      save_path = os.path.join(f'{args.output_dir}/images', f'{fnc:07d}.jpg')
      # crop both eyes and join them in a VGG size image (224x224)
      crop_left  = image[352:(352+224), 347:(347+112)]  
      crop_right = image[352:(352+224), 575:(575+112)]
      crop = np.concatenate((crop_left, crop_right), axis=1) 
      cv2.imwrite(save_path, crop[:, :, ::-1])
      # save the 'w' for later boundary SVM
      save_path = os.path.join(f'{args.output_dir}/w', f'{fnc:07d}.npy') 
      np.save(save_path, w)
      # update counter
      fnc = fnc + 1


  logger.info(f'Saving results.')
  for key, val in results.items():
    if key == 'z' or key == 'wp':
      pass # not interested

if __name__ == '__main__':
  main()
