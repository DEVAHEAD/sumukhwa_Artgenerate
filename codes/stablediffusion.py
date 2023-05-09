import os
from keras_cv.models import StableDiffusion
from PIL import Image

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

    images = model.text_to_image(**options)

    image_paths=[]
    for i,image in enumerate(images):  

        orig_image_path=os.path.join("art_generator\\media\\generatedImages\\",project.projectName+"_"+str(i)+".png")
             
        image.save(orig_image_path) 
        image_paths.append(orig_image_path)
    return images,image_paths

if __name__=="__main__":
    create_ai_image("","")