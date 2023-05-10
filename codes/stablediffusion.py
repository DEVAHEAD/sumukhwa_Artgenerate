import os
from keras_cv.models import StableDiffusion
from PIL import Image
from rembg import remove
import styletransfer

SEED = 119

def create_ai_image(np , pp , project=None):

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
    gray_image = rbg_image.convert('L')
    
    gray_image.save('..\generated_rmbg_gray\sd_output.png', 'png')