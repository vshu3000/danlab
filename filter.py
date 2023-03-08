#!/usr/bin/env python3
import os
import sys
import cv2
from glob import glob
import numpy as np
from mtcnn import MTCNN
#from omegaconf import OmegaConf
#from tensorflow.keras.utils import get_file
#from src.factory import get_model
from config import *

GOOD_DIR = 'filter'
BAD_DIR = 'bad'
os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)
detector = MTCNN()

#weight_file = os.path.join(HOME, 'EfficientNetB3_224_weights.11-3.44.hdf5')
#cfg = OmegaConf.from_dotlist([f"model.model_name=EfficientNetB3", f"model.img_size=224"])
#model = get_model(cfg)
#model.load_weights(weight_file)

for serial, path in enumerate(glob('raw/**/*', recursive=True)):
    print(path)
    try:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        continue
    img_h, img_w = rgb.shape[:2]
    out = detector.detect_faces(rgb)
    good = False
    for _ in [True]:
        if len(out) != 1:
            break
        x1, y1, width, height = out[0]['box']
        if width < MIN_FACE_WIDTH:
            break
        if height < MIN_FACE_HEIGHT:
            break
#        xw1 = max(int(x1 - MARGIN * width), 0)
#        yw1 = max(int(y1 - MARGIN * height), 0)
#        xw2 = min(int(x1 + width + MARGIN * width), img_w)
#        yw2 = min(int(y1 + height + MARGIN * height), img_h)
#        face = cv2.resize(rgb[yw1:yw2, xw1:xw2], (FACE_SIZE, FACE_SIZE))
#
#        faces = face[np.newaxis, :, : , :]
#        results = model.predict(faces)
#        #predicted_genders = results[0]
#        ages = np.arange(0, 101).reshape(101, 1)
#        predicted_ages = results[1].dot(ages).flatten()
#
#        age = predicted_ages[0]

        good = True
    if good:
        cv2.imwrite(os.path.join(GOOD_DIR, '%d.png' % serial), image)
    else:
        cv2.imwrite(os.path.join(BAD_DIR, '%d.png' % serial), image)

