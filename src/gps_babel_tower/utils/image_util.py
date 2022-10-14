import math
import PIL
from typing import List, Optional, Union, Mapping, Tuple


def image_grid(imgs, rows=None, cols=None) -> PIL.Image.Image:
  n_images = len(imgs)
  if not rows and not cols:
    cols = math.ceil(math.sqrt(n_images))
  if not rows:
    rows = math.ceil(n_images / cols)
  if not cols:
    cols = math.ceil(n_images / rows)

  w, h = imgs[0].size
  grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
  grid_w, grid_h = grid.size

  for i, img in enumerate(imgs):
      grid.paste(img, box=(i%cols*w, i//cols*h))
  return grid


def preprocess_image(
  image,
  width=None,
  height=None):
  if isinstance(image, list):
    if isinstance(image[0], Tuple):
      return [(preprocess_image(im), score) for im, score in image]
    else:
      return [preprocess_image(im) for im in image]   
  if isinstance(image, str):
    image = PIL.Image.open(image)
    image = image.convert('RGB')
  if width and height:
    image = image.resize((width, height))
  return image