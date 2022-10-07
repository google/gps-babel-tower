# coding=utf-8
# Copyright 2021 Google LLC..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from gps_babel_tower.tasks.image_generation import ImageGenerator

from google.cloud import storage


def main(args):
  # model is already copied to docker image, no need to download
  if args.use_auth_token:
    use_auth_token = True if args.use_auth_token.lower() == 'true' else args.use_auth_token
    image_generator = ImageGenerator.create(scheduler='lms', use_auth_token=use_auth_token)
  else:
    image_generator = ImageGenerator.create(scheduler='lms', model_path=args.model_path)

  image_size = 512
  text_guidance_scale = 8.2

  client = storage.Client()
  for i in range(args.num_images):
    image = image_generator.text2image(args.prompt,
                                       text_guidance_scale=text_guidance_scale,
                                       width=image_size,
                                       height=image_size)
    if 'gs://' in args.output_path:
      tmp_filename = f'/tmp/image_{i}.png'
      image.save(tmp_filename)
      
      worker_blob = storage.Blob.from_string(f'{args.output_path}/image_{i}.png', client=client)
      print(f'uploading {worker_blob} => {tmp_filename}')
      worker_blob.upload_from_filename(tmp_filename)
    else:
      image.save(os.path.join(args.output_path, f'image_{i}.png'))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Babel Tower Text to Image Example.')
  parser.add_argument('--prompt',
                      required=False,
                      default='A fine painting of a pikachu playing on a bridge, in front of a river, in the style of Vincent van Gogh',
                      help='Text prompt')
  parser.add_argument('--output_path',
                      required=True,
                      help='an integer for the accumulator')
  parser.add_argument('--num_images',
                      required=False,
                      default=3,
                      type=int,
                      help='Number of images to generate')
  parser.add_argument('--model_path',
                      required=False,
                      default='/models/stable-diffusion-v1-4',
                      help='path to stable diffusion model')
  parser.add_argument('--use_auth_token',
                      required=False,
                      default='')
  args, _ = parser.parse_known_args()
  main(args)