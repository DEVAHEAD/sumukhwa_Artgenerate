# 일부 주석처리 된 것은 stable diffusion 결과를 8개 받았을 경우, 사용하는 코드
# demo 실행 속도와 결과물의 질을 고려하여, demo 코드에서는 하나의 stable diffusion 결과물만 이용함.

import stablediffusion
import styletransfer
import os
import random
from shutil import copyfile

from PIL import Image

import tensorflow.compat.v2 as tf

import tensorflow as tf
from tensorflow import keras
from keras.applications import imagenet_utils
from keras.applications.vgg19 import VGG19

import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from art_generator import model

neg_p = sys.argv[1]
pos_p = sys.argv[2]
project_id = sys.argv[3]
type = sys.argv[4]

project = model.Projects.objects.get(id=project_id)

# for i in range(7):
#    sd_result = stablediffusion.create_ai_image(neg_p, pos_p, project)
sd_result = stablediffusion.create_ai_image(neg_p, pos_p, project)
sd_images = os.listdir('..\generated_rmbg_gray')

# Generated image size
RESIZE_HEIGHT = 607

NUM_ITER = 3000

# Weights of the different loss components
CONTENT_WEIGHT = 8e-4
STYLE_WEIGHT = 8e-1

# The layer to use for the content loss.
CONTENT_LAYER_NAME = "block5_conv2"

# List of layers to use for the style loss.
STYLE_LAYER_NAMES = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

def random_target_list(n, type):
    if type == 1: #백묘법
        ldir = os.listdir('..\sample_target dataset_ST1')
    if type == 2: #구륵법
        ldir = os.listdir('..\sample_target dataset_ST2')
    if type == 3: #몰골법
        ldir = os.listdir('..\sample_target dataset_ST3')
    selected = random.sample(ldir, n)
    return selected

target_list = random_target_list(3, type)
img = np.array(target_list)

# Prepare content, stlye images
path = os.path.abspath(os.getcwd())
# content_image_path = np.array(sd_images)
content_image_path = sd_images[0]
style_image_path1 = img[0]
style_image_path2 = img[1]
style_image_path3 = img[2]

# arr_result = [[0 for j in range(1)] for i in range(7)]

#for i in range(7):
#    result_height, result_width = styletransfer.get_result_image_size(content_image_path[i], RESIZE_HEIGHT)
#    arr_result[i] = [result_height, result_width]

result_height, result_width = styletransfer.get_result_image_size(content_image_path, RESIZE_HEIGHT)

# Preprocessing

def preprocess_image(image_path, target_height, target_width):
    img = keras.preprocessing.image.load_img(image_path, target_size = (target_height, target_width))
    arr = keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis = 0)
    arr = styletransfer.preprocess_input(arr)
    return tf.convert_to_tensor(arr)

# tensor = []: 각각의 content image에 대한 tensor 값을 다차원으로 저장할 배열 선언

#for i in range(7):
#    content_tensor = preprocess_image(content_image_path, result_height, result_width)
#    tensor: [[preprocess_image's return array, index number], ---] 꼴로 저장될 다차원 배열
content_tensor = preprocess_image(content_image_path, result_height, result_width)
style_tensor1 = preprocess_image(style_image_path1, result_height, result_width)
style_tensor2 = preprocess_image(style_image_path2, result_height, result_width)
style_tensor3 = preprocess_image(style_image_path3, result_height, result_width)
generated_image = tf.Variable(tf.random.uniform(style_tensor1.shape, dtype=tf.dtypes.float32))

def get_model():
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = VGG19(weights = 'imagenet', include_top = False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in VGG19 (as a dict).
    return keras.Model(inputs = model.inputs, outputs = outputs_dict)

modelst = get_model()
optimizer = styletransfer.get_optimizer()

#for i in range(7):
#    content_features = modelst(content_tensor[i])
#    style_features1 = modelst(style_tensor1)
#    style_features2 = modelst(style_tensor2)
#    style_features3 = modelst(style_tensor3)
#    img_result = styletransfer.deprocess_image(generated_image, result_height, result_width)

content_features = modelst(content_tensor)
style_features1 = modelst(style_tensor1)
style_features2 = modelst(style_tensor2)
style_features3 = modelst(style_tensor3)

def save_result(generated_image, result_height, result_width, name):
    img = styletransfer.deprocess_image(generated_image, result_height, result_width)
    keras.preprocessing.image.save_img(name, img)

img = styletransfer.deprocess_image(generated_image, result_height, result_width)

a = [style_features1, style_features2, style_features3]
for iter in range(NUM_ITER):
    if iter%1000 == 0:
      i = int(iter/1000)
      style = a[i]
    with tf.GradientTape() as tape:
        loss = styletransfer.compute_loss(model, generated_image, content_features, style)

    grads = tape.gradient(loss, generated_image)

    optimizer.apply_gradients([(grads, generated_image)])
    
    if iter == 2999:
        name = "/content/drive/MyDrive/Colab Notebooks/수묵화/result/generated_at_iteration_%d.png" % (iter + 1)
        save_result(generated_image, result_height, result_width, name)
