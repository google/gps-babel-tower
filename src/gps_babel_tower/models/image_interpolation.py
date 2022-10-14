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

import inspect
from typing import List, Optional, Union, Tuple

import numpy as np
import torch

from PIL import Image
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""
    # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def save_pil_image(image, path):
    image.save(path)
    return Path(path)
  
def make_scheduler(num_inference_steps, from_scheduler=None):
    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
    scheduler.set_timesteps(num_inference_steps, offset=1)
    if from_scheduler:
        scheduler.cur_model_output = from_scheduler.cur_model_output
        scheduler.counter = from_scheduler.counter
        scheduler.cur_sample = from_scheduler.cur_sample
        scheduler.ets = from_scheduler.ets[:]
    return scheduler


class InterpolationPipeline(DiffusionPipeline):
    """
    From https://github.com/huggingface/diffusers/pull/241
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt_start: str,
        prompt_end: str,
        interp_point: List[float]=[0.5],
        width: int=512,
        height: int=512,
        num_inference_steps: int=50,
        prompt_strength: float = 0.8,
        guidance_scale: float = 7.5,
        generator=None
    ) -> Image:
        batch_size = 1

        # Generate initial latents to start to generate animation frames from
        initial_scheduler = self.scheduler = make_scheduler(
            num_inference_steps
        )
        num_initial_steps = int(num_inference_steps * (1 - prompt_strength))
        print(f"Generating initial latents for {num_initial_steps} steps")
        initial_latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device="cuda",
        )
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings_start = self.embed_text(
            prompt_start, do_classifier_free_guidance, batch_size
        )
        text_embeddings_end = self.embed_text(
            prompt_end, do_classifier_free_guidance, batch_size
        )
        text_embeddings_mid = slerp(0.5, text_embeddings_start, text_embeddings_end)
        latents_mid = self.denoise(
            latents=initial_latents,
            text_embeddings=text_embeddings_mid,
            t_start=1,
            t_end=num_initial_steps,
            guidance_scale=guidance_scale,
        )

        print("Generating first frame")
        # re-initialize scheduler
        self.scheduler = make_scheduler(num_inference_steps, initial_scheduler)
        latents_start = self.denoise(
            latents=latents_mid,
            text_embeddings=text_embeddings_start,
            t_start=num_initial_steps,
            t_end=None,
            guidance_scale=guidance_scale,
        )
        image_start = self.latents_to_image(latents_start)

        print("Generating last frame")
        # re-initialize scheduler
        self.scheduler = make_scheduler(num_inference_steps, initial_scheduler)
        latents_end = self.denoise(
            latents=latents_mid,
            text_embeddings=text_embeddings_end,
            t_start=num_initial_steps,
            t_end=None,
            guidance_scale=guidance_scale,
        )
        # print('latents end:', latents_end.dtype)
        image_end = self.latents_to_image(latents_end)

        # Generate latents for animation frames
        frames_latents = []
        images = [image_start]
        for t in interp_point:
          print(f'Generating interpolated image at {t}')
          text_embeddings = slerp(
              t,
              text_embeddings_start,
              text_embeddings_end,
          )

          # re-initialize scheduler
          self.scheduler = make_scheduler(
              num_inference_steps, initial_scheduler
          )

          latents = self.denoise(
              latents=latents_mid,
              text_embeddings=text_embeddings,
              t_start=num_initial_steps,
              t_end=None,
              guidance_scale=guidance_scale,
          )

          image = self.latents_to_image(latents)
          images.append(image)
        images.append(image_end)
        return [self.numpy_to_pil(image)[0] for image in images]

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def denoise(self, latents, text_embeddings, t_start, t_end, guidance_scale):
        do_classifier_free_guidance = guidance_scale > 1.0

        for i, t in tqdm(list(enumerate(self.scheduler.timesteps[t_start:t_end]))):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        return latents

    def embed_text(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool,
        batch_size: int,
    ) -> torch.FloatTensor:
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def latents_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.autocast("cuda"):
          image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    def safety_check(self, image):
        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)
        _, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values
        )
        if has_nsfw_concept[0]:
            raise Exception("NSFW content detected, please try a different prompt and/or seed")
