import os
from keras_cv.models import StableDiffusion
from PIL import Image
import cv2
from rembg import remove

SEED = 119

def create_ai_image(np , pp ,project=None, limit=10):

    if not project:
        class Project:
            projectName="test_default" 
        project=Project()

    model = StableDiffusion(img_height=512, img_width=512, jit_compile=False)

    if np!="" and pp!="":
        positive = pp
        negative = np
    else:
        positive = "a photograph of an astronaut riding a horse"
        negative = "not a red horse"

    options = dict(
        prompt = positive,
        negative_prompt = negative,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7,
        seed=SEED
    )

    sd_image = model.text_to_image(**options)
    rbg_image = remove(sd_image)
    gray_image = cv2.cvtColor(rbg_image, cv2.COLOR_BGR2GRAY)
    
    gray_image.save('C:\Users\Soyeun_2\Desktop\DEVs\sumukhwa\sumukhwa_Artgenerate\generated_rmbg_gray\gray.png', 'png')