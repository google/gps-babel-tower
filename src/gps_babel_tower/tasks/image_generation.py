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

from typing import List, Optional, Union, Mapping
import torch
from gps_babel_tower.models.diffusion import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import PIL


class ImageGenerator:
  def __init__(self, pipe=None, generator=None):
    self.pipe = pipe
    # random number generator for reproducibility
    self.generator = None   

  @classmethod
  def create(cls,
             pipe=None,
             model_path: str ='CompVis/stable-diffusion-v1-4',
             manual_seed: Optional[int] = None,
             scheduler = None,
             use_auth_token = False,
             device='cuda'):
    if manual_seed:
      generator = torch.cuda.manual_seed(manual_seed)
    else:
      generator = None

    if not pipe:
      # Create pipeline
      pipe_config = dict(
        revision="fp16",
        torch_dtype=torch.float16
      )

      if scheduler == 'lms':
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        pipe_config['scheduler'] = scheduler

      if use_auth_token:
        pipe_config['use_auth_token'] = use_auth_token

      print('Loading pipeline', pipe_config)
      pipe = StableDiffusionPipeline.from_pretrained(model_path, **pipe_config).to('cuda')
      print('Pipeline loaded')

    return cls(pipe=pipe, generator=generator)

  def text2image(self, prompt, **kwargs):
    with torch.autocast("cuda"):
      images = self.pipe(prompt=prompt, generator=self.generator, **kwargs).images
      if len(images) == 1:
        images = images[0]
      return images
    
  def text_guided_image2image(self, prompt, init_image, redraw_strength=0.8, text_guidance_scale=7.5, **kwargs):
    """Text guided image to image generation.
    
    Args:
      init_image:  The initial image to start.
      prompt: The text to guide the generation.
      redraw_strength:  0-1 value indicating how hard to redraw the image. 0 means no modifications to initial image at all, while 1 means redraw from scratch
      text_guidance_scale:  1 means no guidance. The bigger the value, the more relevant the image is to the text.
    """
    with torch.autocast("cuda"):
      images = self.pipe(prompt=prompt,
                         generator=self.generator,
                         init_image=init_image,
                         strength=redraw_strength,
                         guidance_scale=text_guidance_scale,
                         **kwargs).images
      if len(images) == 1:
        images = images[0]
      return images
  
  def similar_image2image(self, init_image, redraw_strength=0.7, **kwargs):
    with torch.autocast("cuda"):
      images = self.pipe(prompt='trending on instagram',
                         generator=self.generator,
                         init_image=init_image,
                         strength=redraw_strength,
                         guidance_scale=7.5,
                         **kwargs).images
      if len(images) == 1:
        images = images[0]
      return images

  def image_inpaint(self,
                    prompt,
                    init_image,
                    mask_image,
                    redraw_strength=0.75,
                    text_guidance_scale=7.5,
                    **kwargs):
    with torch.autocast("cuda"):
      images = self.pipe(prompt=prompt,
                         generator=self.generator,
                         init_image=init_image,
                         mask_image=mask_image,
                         strength=redraw_strength,
                         guidance_scale=text_guidance_scale,
                         **kwargs).images
      if len(images) == 1:
        images = images[0]
      return images

  def image_interpolation(self,
                          mixed_image: Mapping[str, Union[str, PIL.Image.Image]],
                          width=None,
                          height=None):
    with torch.autocast("cuda"):
      latents = self.pipe.encode_mixed_image(mixed_image=mixed_image,
                                             encode_type='sample',
                                             generator=self.generator,
                                             width=width,
                                             height=height)
      images = self.pipe.decode_latents(latents)

    if len(images) == 1:
        images = images[0]
    return images