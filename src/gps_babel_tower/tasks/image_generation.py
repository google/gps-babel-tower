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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import LMSDiscreteScheduler
import PIL

from gps_babel_tower.utils.image_util import preprocess_image
from gps_babel_tower.models.image_interpolation import InterpolationPipeline
from gps_babel_tower.models.image2image import StableDiffusionImageEmbedPipeline


class Text2ImageGenerator:
  def __init__(self, pipe=None, seed=None, safety_check=False):
    # random seed for reproducibility
    self.seed = seed
    
    pipe_config_keys = [
      'vae', 'tokenizer', 'unet', 'scheduler', 'text_encoder',
      'safety_checker', 'feature_extractor',
    ]
    pipe_config_dict = {
      k: getattr(pipe, k) for k in pipe_config_keys
    }
    
    if not safety_check:
      pipe_config_dict['safety_checker'] = None
      pipe_config_dict['feature_extractor'] = None
      
    self.pipe_config_dict = pipe_config_dict
    self.pipe = pipe
    
  @property
  def text2image_pipe(self):
    if not hasattr(self, '_text2image_pipe'):
      print('Init StableDiffusionPipeline')
      self._text2image_pipe = StableDiffusionPipeline(**self.pipe_config_dict)
    return self._text2image_pipe
  
  @property
  def image2image_pipe(self):
    if not hasattr(self, '_image2image_pipe'):
      print('Init StableDiffusionImg2ImgPipeline')
      self._image2image_pipe = StableDiffusionImg2ImgPipeline(**self.pipe_config_dict)
    return self._image2image_pipe
  
  @property
  def inpaint_pipe(self):
    if not hasattr(self, '_inpaint_pipe'):
      print('init StableDiffusionInpaintPipeline')
      self._inpaint_pipe = StableDiffusionInpaintPipeline(**self.pipe_config_dict)
    return self._inpaint_pipe
  
  @property
  def interpolation_pipe(self):
    if not hasattr(self, '_interpolation_pipe'):
      print('init StableDiffusionInpaintPipeline')
      self._interpolation_pipe = InterpolationPipeline(**self.pipe_config_dict)
    return self._interpolation_pipe
  
  @property
  def generator(self):
    if self.seed:
      return torch.cuda.manual_seed(self.seed)
    return None

  @classmethod
  def instance(
    cls,
    pipe=None,
    model_path: str ='CompVis/stable-diffusion-v1-4',
    manual_seed: Optional[int] = None,
    scheduler = None,
    use_auth_token = False,
    device='cuda',
    safety_check=False,
    force_create_new_pipe=False):
    
    if force_create_new_pipe or (not hasattr(cls, '_pipe')):
      # Create pipeline
      pipe_config = dict(
        revision="fp16",
        torch_dtype=torch.float16
      )

      if scheduler == 'lms':
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

      if scheduler:
        pipe_config['scheduler'] = scheduler

      if use_auth_token:
        pipe_config['use_auth_token'] = use_auth_token

      print('Loading pipeline', pipe_config)
      cls._pipe = StableDiffusionPipeline.from_pretrained(model_path, **pipe_config).to('cuda')
      print('Pipeline loaded')

    return cls(pipe=cls._pipe, seed=manual_seed, safety_check=safety_check)

  def text2image(self, prompt, **kwargs):
    with torch.autocast("cuda"):
      images = self.text2image_pipe(prompt=prompt, generator=self.generator, **kwargs).images
      if len(images) == 1:
        images = images[0]
      return images
    
  def text_guided_image2image(self, prompt, init_image, width=None, height=None, **kwargs):
    with torch.autocast("cuda"):
      images = self.image2image_pipe(
        prompt=prompt,
        init_image=preprocess_image(init_image, width=width, height=height),
        generator=self.generator,
        **kwargs).images
      if len(images) == 1:
        images = images[0]
      return images

  def image_inpaint(self,
                    prompt,
                    init_image,
                    mask_image,
                    **kwargs):
    with torch.autocast("cuda"):
      images = self.inpaint_pipe(
        prompt=prompt,
        generator=self.generator,
        init_image=preprocess_image(init_image),
        mask_image=preprocess_image(mask_image),
        **kwargs).images
      if len(images) == 1:
        images = images[0]
      return images
    
  def image_interpolation(self, **kwargs):
    with torch.autocast("cuda"):
      images = self.interpolation_pipe(**kwargs)
      return images
    
  

class Image2ImageGenerator:
  def __init__(self, pipe=None, seed=None, safety_check=False):
    # random seed for reproducibility
    self.seed = seed
    
    pipe_config_keys = [
      'vae', 'image_encoder', 'unet', 'scheduler',
      'safety_checker', 'feature_extractor',
    ]
    pipe_config_dict = {
      k: getattr(pipe, k) for k in pipe_config_keys
    }
    
    if not safety_check:
      pipe_config_dict['safety_checker'] = None
      pipe_config_dict['feature_extractor'] = None
      
    self.pipe_config_dict = pipe_config_dict
    self.pipe = StableDiffusionImageEmbedPipeline(**pipe_config_dict)
    
  @property
  def generator(self):
    if self.seed:
      return torch.cuda.manual_seed(self.seed)
    return None
  
  @classmethod
  def instance(
    cls,
    model_path: str ='lambdalabs/sd-image-variations-diffusers',
    manual_seed: Optional[int] = None,
    scheduler = None,
    use_auth_token = False,
    device='cuda',
    safety_check=False,
    force_create_new_pipe=False):
    if force_create_new_pipe or (not hasattr(cls, '_pipe')):
      pipe_config = {}
      if scheduler == 'lms':
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

      if scheduler:
        pipe_config['scheduler'] = scheduler

      if use_auth_token:
        pipe_config['use_auth_token'] = use_auth_token
      
      cls._pipe = StableDiffusionImageEmbedPipeline.from_pretrained(model_path, **pipe_config).to(device)
    return Image2ImageGenerator(pipe=cls._pipe, seed=manual_seed, safety_check=safety_check)
  
  def generate_similar_image(self, image, width=512, height=512, **kwargs):
    if isinstance(image, Mapping):
      image = list(image.items())
    image = preprocess_image(image, width, height)
    results = self.pipe(image, generator=self.generator, width=width, height=height, **kwargs).images
    if len(results) == 1:
      results = results[0]
    return results