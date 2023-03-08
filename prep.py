#!/usr/bin/python3
import os
import sys
HOME=os.path.dirname(__file__)
sys.path.append(os.path.join(HOME, 'stable-diffusion-webui'))
import torch
from modules import devices
from modules.textual_inversion.preprocess import preprocess_work
from modules.shared import interrogator

#src = 'smooth/restored_imgs'
src = 'filter'
dst = 'samples/100_images'

os.makedirs(dst, exist_ok=True)

devices.dtype = torch.float16
interrogator.load()
print(interrogator.dtype)

preprocess_work(src, dst, 512, 512, 'append', True, False, True, False)

