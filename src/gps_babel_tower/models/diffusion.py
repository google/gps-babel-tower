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
import warnings
from typing import List, Optional, Union, Mapping

import torch
import numpy as np

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

import PIL

from torchvision import transforms
to_tensor_tfm = transforms.ToTensor()


class StableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            warnings.warn(
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file",
                DeprecationWarning,
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)
        
    def create_latents(
        self,
        height: int = 512,
        width: int = 512,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
    ):
        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_device = "cpu" if self.device.type == "mps" else self.device
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=latents_device,
        )
        return latents

    def encode_text(
        self,
        text: Union[str, List[str]],
    ):
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # (batch_size, max_len, hidden) 
        return text_embeddings
    
    def encode_mixed_text(
        self,
        mixed_text: Mapping[str, float]
    ):
        prompts, scores = list(zip(*mixed_text.items()))
        scores_tensor = torch.tensor(scores).to(self.device)
        normalized_scores =  scores_tensor / torch.sum(scores_tensor)
        # (batch_size, seq_len, hidden)
        embeddings = self.encode_text(prompts)
        mixed_embeddings = (normalized_scores.unsqueeze(-1).unsqueeze(-1) * embeddings).sum(0, keepdim=True)
        return mixed_embeddings
    
    def decode_latents(
        self,
        latents: torch.FloatTensor,
        output_type: Optional[str] = "pil",
    ):
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)
            
        return image
    
    def encode_image(
        self,
        image: Union[str, PIL.Image.Image],
        encode_type: str='mode',
        generator: Optional[torch.Generator] = None,
        width=None,
        height=None,
    ):
        if isinstance(image, str):
            # Load image from file
            image = PIL.Image.open(image).convert('RGB')
            
        # resize to integer multiple of 32
        if width and height:
            w, h = width, height
        else:
            w, h = image.size
            w, h = map(lambda x: x - x % 32, (w, h))
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        
        # convert from RGB image (batch_size, w, h, channel) value range(0, 255)
        # to shape (batch_size, channel, w, h) value range (-1, 1)
        image = to_tensor_tfm(image)[None] * 2.0 - 1.0
        
        with torch.no_grad():
            latent_dist = self.vae.encode(image.to(self.device)).latent_dist

        if encode_type == 'mode':
            latent = latent_dist.mode()
        elif encode_type == 'mean':
            latent = latent_dist.mean()
        elif encode_type == 'sample':
            latent = latent_dist.sample(generator=generator)

        return 0.18215 * latent
      
    def encode_mixed_image(
        self,
        mixed_image: Mapping[str, Union[str, PIL.Image.Image]],
        **kwargs
    ):
        images, scores = list(zip(*mixed_image.items()))
        scores_tensor = torch.tensor(scores).to(self.device)
        normalized_scores =  scores_tensor / torch.sum(scores_tensor)
        # (batch_size, seq_len, hidden)
        embeddings = torch.cat([self.encode_image(image, **kwargs) for image in images])
        mixed_embeddings = (normalized_scores[:,None,None,None] * embeddings).sum(0, keepdim=True)
        return mixed_embeddings
    
    def preprocess_mask(
        self,
        mask: Union[str, torch.FloatTensor, PIL.Image.Image],
        width=None,
        height=None,
    ):
        if isinstance(mask, torch.FloatTensor):
            # raw mask
            return mask

        if isinstance(mask, str):
            # Load image from file
            mask = PIL.Image.open(mask)
        
        mask = mask.convert("L")
        w, h = width, height
        if not w or not h:
          w, h = mask.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0 # shape: (w//8, h//8)
        mask = np.tile(mask, (self.unet.in_channels, 1, 1)) # shape: (4, w//8, h//8), same as latents shape because mask needs to multiply with latents
        mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask)
        return mask.to(self.device)
    
    def mask_to_image(
        self,
        mask: torch.FloatTensor
    ):
        mask = mask[0, 0:3].permute(1,2,0).cpu().numpy()
        mask = (mask * 255).astype('uint8')
        return PIL.Image.fromarray(mask)
    
    def add_noise(
        self,
        num_inference_steps: int,
        strength: float = 0.8,
        latents: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        batch_size: int = 1,
        noise: Optional[torch.FloatTensor] = None,
    ):
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            timesteps = torch.tensor(
                [num_inference_steps - init_timestep] * batch_size, dtype=torch.long, device=self.device
            )
        else:
            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)
        # add noise to latents using the timesteps
        latents = self.scheduler.add_noise(latents, noise, timesteps)
        latents = latents.float()  # fix latent type changed to double bug
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str], Mapping[str, float]],
        init_image: Optional[Union[str, PIL.Image.Image, Mapping[str, Union[str, PIL.Image.Image] ]]]=None,
        mask_image: Optional[Union[str, torch.FloatTensor, PIL.Image.Image]]=None,
        strength: float = 0.8,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        safety_check: bool = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            safety_check (`bool`, *optional*, defaults to `Fasle`):
                do safety check.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)
        
        is_mixed_prompt = False
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, Mapping):
            # mixed prompt with scores
            batch_size = 1
            is_mixed_prompt = True
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        if is_mixed_prompt:
            text_embeddings = self.encode_mixed_text(prompt)
        else:
            text_embeddings = self.encode_text(prompt)            

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_embeddings = self.encode_text([""] * batch_size)
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        
        if init_image:
            # start from the init_image, and add some random noise
            if isinstance(init_image, Mapping):
                latents = self.encode_mixed_image(mixed_image=init_image,
                                                  encode_type='sample',
                                                  generator=generator,
                                                  width=width,
                                                  height=height)
            else:
                latents = self.encode_image(image=init_image,
                                            encode_type='sample',
                                            generator=generator,
                                            width=width,
                                            height=height)
            latents = torch.cat([latents] * batch_size)
            init_latents_orig = latents
            noise = torch.randn(latents.shape, generator=generator, device=self.device)
            latents = self.add_noise(
                num_inference_steps=num_inference_steps,
                strength=strength,
                latents=latents,
                batch_size=batch_size,
                generator=generator,
                noise=noise
            )
        else: 
            if latents is not None:
                latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
                if latents.shape != latents_shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            else:
                # use random noise
                latents = self.create_latents(batch_size=batch_size,
                                              width=width,
                                              height=height,
                                              generator=generator)
            # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = latents * self.scheduler.sigmas[0]
        latents = latents.to(self.device)
        
        if mask_image is not None:
            mask = self.preprocess_mask(mask_image, width=width, height=height)
            mask = torch.cat([mask] * batch_size)
            # check sizes
            if not mask.shape == latents.shape:
                raise ValueError("The mask and init_image should be the same size!")

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        if init_image:
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)
            t_start = max(num_inference_steps - init_timestep + offset, 0)
        else:
            t_start = 0
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps[t_start:])):
            t_index = t_start + i
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[t_index]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, t_index, latents, **extra_step_kwargs).prev_sample
                if mask_image is not None:
                    # masking
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor(t_index))
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                if mask_image is not None:
                    # masking
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)
                
            if mask_image is not None:
                init_latents_proper = init_latents_proper.float()  # fix strange float64 bug
                latents = (init_latents_proper * mask) + (latents * (1 - mask))
        
        image = self.decode_latents(latents, output_type='raw')
        
        if safety_check:
            # run safety checker
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values)
        else:
            has_nsfw_concept = False

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
__all__ = ['StableDiffusionPipeline']