### Imports
import os
from os.path import isfile, join
import time

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

#%% Tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)

import tensorflow as tf

# Evita o erro "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Verifica se a GPU está disponível:
print(tf.config.list_physical_devices('GPU'))
# Verifica se a GPU está sendo usada na sessão
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(sess)

#%%

### Prepara as pastas
dataset_folder = 'C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'

train_folder = dataset_folder+'train/'
test_folder = dataset_folder+'val/'

result_folder = 'results-train-autoencoder/'
result_test_folder = 'results-test-autoencoder/'

model_folder = 'model/'

if not os.path.exists(result_folder):
    os.mkdir(result_folder)
    
if not os.path.exists(result_test_folder):
    os.mkdir(result_test_folder)
    
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
checkpoint_dir = 'training_checkpoints_autoencoder'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

## HIPERPARÂMETROS
LAMBDA = 100
BATCH_SIZE = 1
BUFFER_SIZE = 200
IMG_SIZE = 256

EPOCHS = 5
CHECKPOINT_EPOCHS = 1
LOAD_CHECKPOINT = True
FIRST_EPOCH = 1
NUM_TEST_PRINTS = 1

# Controla se vai usar o gerador completo ou o modo encoder/decoder
USE_FULL_GENERATOR = True

#%% FUNÇÕES DE APOIO

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    return image

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

def random_crop(input_image):
    cropped_image = tf.image.random_crop(value = input_image, size = [IMG_SIZE, IMG_SIZE, 3])
    return cropped_image

# normalizing the images to [-1, 1]
def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

# Equivalente a random_jitter = tf.function(random.jitter)
@tf.function()
def random_jitter(input_image):
    # resizing to 286 x 286 x 3
    input_image = resize(input_image, 286, 286)
    
    # randomly cropping to 256 x 256 x 3
    input_image = random_crop(input_image)
    
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
    
    return input_image

def load_image_train(image_file):
    input_image = load(image_file)    
    input_image = random_jitter(input_image)
    input_image = normalize(input_image)
    return input_image

def load_image_test(image_file):
    input_image = load(image_file)    
    input_image = resize(input_image, IMG_SIZE, IMG_SIZE)
    input_image = normalize(input_image)
    return input_image

def generate_images(encoder, decoder, img_input):
    latent = encoder(img_input, training=True)
    img_predict = decoder(latent, training=True)
    plt.figure(figsize=(15,15))
    
    display_list = [img_input[0], img_predict[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()  
    
def generate_save_images(encoder, decoder, img_input, save_destination, filename):
    latent = encoder(img_input, training=True)
    img_predict = decoder(latent, training=True)
    f = plt.figure(figsize=(15,15))
    
    print("Latent Vector:")
    print(latent)
    
    display_list = [img_input[0], img_predict[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    
    f.savefig(save_destination + filename)
    
def generate_save_images_gen(generator, img_input, save_destination, filename):
    img_predict = generator(img_input, training=True)
    f = plt.figure(figsize=(15,15))
    
    display_list = [img_input[0], img_predict[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    
    f.savefig(save_destination + filename)


#%% BLOCOS RESNET

def resnet_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet baseado na Resnet34
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x


def resnet_block_transpose(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet baseado na Resnet34, mas invertido
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


def resnet_bottleneck_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet bottleneck, baseado na Resnet50
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Terceira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x

def resnet_downsample_bottleneck_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet bottleneck, com redução de dimensão, baseado na Resnet50
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Convolução da skip connection
    skip = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)
    
    # Primeira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Terceira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x

def upsample(x, filters):
    
    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN
    
    initializer = tf.random_normal_initializer(0., 0.02)
    
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3) , strides = (2, 2), padding = "same",
                               kernel_initializer=initializer, use_bias = True)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x


def downsample(x, filters):
    
    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN

    initializer = tf.random_normal_initializer(0., 0.02)
    
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3) , strides = (2, 2), padding = "same",
                               kernel_initializer=initializer, use_bias = True)(x)
    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x


#%% MODELOS

def cyclegan_resnet_encoder(apply_batchnorm = True, apply_dropout=False):
    
    '''
    Adaptado no gerador utilizado no paper CycleGAN
    '''
    
    # Inicializa a rede
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)

    # Camadas finais
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = 512, activation = 'softmax')(x)    

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def cyclegan_resnet_decoder():
    
    '''
    Adaptado no gerador utilizado no paper CycleGAN
    '''
    
    # Inicializa a rede
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape = [512])
    x = tf.expand_dims(inputs, axis = 1)
    x = tf.expand_dims(x, axis = 1)
    
    # Reconstrução da imagem
    x = upsample(x, 512)
    x = upsample(x, 512)
    x = upsample(x, 512)
    x = upsample(x, 512)
    x = upsample(x, 256)
    x = upsample(x, 128)
    x = upsample(x, 64)
    
    # Última camada
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)    


def cyclegan_resnet_generator(apply_batchnorm = True, apply_dropout=False):
    
    '''
    Adaptado no gerador utilizado no paper CycleGAN
    '''
    
    # Inicializa a rede
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
        
    # Reconstrução da imagem
    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    x = tf.keras.layers.ZeroPadding2D([[2, 2],[2, 2]])(x)
    x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (7, 7) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    #print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def cyclegan_resnet_adapted_generator(apply_batchnorm = True, apply_dropout=False):
    
    '''
    Adaptado no gerador utilizado no paper CycleGAN
    '''
    
    # Inicializa a rede
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
        
    # Criação do vetor latente
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Dense(units = 512, kernel_initializer=initializer)(x)
    
    # Criação do vetor latente (alternativa)
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Reconstrução da imagem
    for i in range(5):
        x = tf.keras.layers.Conv2DTranspose(filters = vecsize, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
    x = tf.keras.layers.Conv2DTranspose(filters = vecsize/2, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = vecsize/4, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = vecsize/8, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[2, 2],[2, 2]])(x)
    x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (7, 7) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    # print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)

#%%

def VT_full_resnet_generator(apply_batchnorm = True, apply_dropout=False):

    
    # Inicializa a rede
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
    
    # Criação do vetor latente 
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Upsamplings
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block_transpose(x, 256)
    
    # Reconstrução pós blocos residuais
    
    #--
    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    #--
    x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[1, 1],[1, 1]])(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (7, 7) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    # print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)

#vtmodel = VT_full_resnet_generator()
#vtmodel.save("teste_vtmodel.h5")


#%%

def discriminator():
    
    '''
    Adaptado do discriminador utilizado no paper CycleGAN
    '''
    
    # Inicializa a rede e os inputs
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    
    # Camadas intermediárias
    x = downsample(x, 64) # (bs, 128, 128, 64)
    x = downsample(x, 128) # (bs, 64, 64, 128)
    x = downsample(x, 256) # (bs, 32, 32, 256)
    x = tf.keras.layers.ZeroPadding2D()(x) # (bs, 34, 34, 256)
    
    # Camada de ajuste
    x = tf.keras.layers.Conv2D(512, 3, strides=1, kernel_initializer=initializer)(x) # (bs, 31, 31, 512)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x) # (bs, 33, 33, 512)
    
    # Camada final (30 x 30 x 1) - Para usar o L1 loss
    x = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer)(x) # (bs, 30, 30, 1)
    
    return tf.keras.Model(inputs=[inp, tar], outputs=x)


def patchgan_discriminator():
    
    '''
    Adaptado diretamente do PatchGAN usado nos papers da CycleGAN e Pix2Pix
    '''
    
    # Inicializa a rede e os inputs
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    # Na implementação em torch, a concatenação ocorre dentro da classe pix2pixmodel
    x = tf.keras.layers.concatenate([inp, tar]) 
    
    # Convoluções
    x = tf.keras.layers.Conv2D(64, 4, strides=2, kernel_initializer=initializer, padding = 'same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(128, 4, strides=2, kernel_initializer=initializer, padding = 'valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(256, 4, strides=2, kernel_initializer=initializer, padding = 'valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, padding = 'same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    # Camada final (30 x 30 x 1) - Para usar o L1 loss
    x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, padding = 'same')(x)
    
    return tf.keras.Model(inputs=[inp, tar], outputs=x)

#%% DEFINIÇÃO DAS LOSSES

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


'''
Perda do gerador
É uma perda de entropia cruzada sigmóide das imagens geradas e uma série de algumas .
O documento também inclui a perda de L1 que é MAE (erro médio absoluto) entre a imagem gerada e a imagem alvo.
Isso permite que a imagem gerada se torne estruturalmente semelhante à imagem de destino.
A fórmula para calcular a perda total do gerador = gan_loss + LAMBDA * l1_loss, onde LAMBDA = 100. Este valor foi decidido pelos autores do artigo .
O procedimento de treinamento para o gerador é mostrado abaixo:
'''

def generator_loss(disc_generated_output, gen_output, target):
    
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss


'''
Perda de discriminador

A função de perda do discriminador leva 2 entradas; imagens reais, imagens geradas
real_loss é uma perda de entropia cruzada sigmóide das imagens reais e uma matriz de uns (uma vez que estas são as imagens reais)
Generated_loss é uma perda de entropia cruzada sigmóide das imagens geradas e uma matriz de zeros (uma vez que estas são as imagens falsas)
Então o total_loss é a soma de real_loss e o generated_loss
'''

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss


#%% FUNÇÕES DO TREINAMENTO

'''
Para cada exemplo de entrada, gere uma saída.
O discriminador recebe a input_image e a imagem gerada como a primeira entrada. A segunda entrada é input_image e target_image.
Em seguida, calculamos a perda do gerador e do discriminador.
Em seguida, calculamos os gradientes de perda em relação às variáveis ​​do gerador e do discriminador (entradas) e os aplicamos ao otimizador.
Em seguida, registre as perdas no TensorBoard.
'''

# Função para o encoder/decoder
@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
        
        latent = encoder(input_image, training = True)
        gen_image = decoder(latent, training = True) 
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)
          
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_gen, gen_image, target)
        disc_loss = discriminator_loss(disc_real, disc_gen)
    
    encoder_gradients = enc_tape.gradient(gen_total_loss, encoder.trainable_variables)
    decoder_gradients = dec_tape.gradient(gen_total_loss, decoder.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
    decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss)

# Função para o gerador
@tf.function
def train_step_gen(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        gen_image= generator(input_image, training = True)
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)
          
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_gen, gen_image, target)
        disc_loss = discriminator_loss(disc_real, disc_gen)
    
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss)

    
             
'''
O loop de treinamento real:

Repete o número de épocas.
Em cada época, ele limpa a tela e executa generate_images para mostrar seu progresso.
Em cada época, ele itera sobre o conjunto de dados de treinamento, imprimindo um '.' para cada exemplo.
Ele salva um ponto de verificação a cada 20 épocas.
'''

# Função para o encoder/decoder
def fit(train_ds, first_epoch, epochs, test_ds):
    
    t0 = time.time()
    
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            generate_save_images(encoder, decoder, example_input, result_folder, filename)
        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            target = input_image
            losses = train_step(input_image, target, epoch)
            
            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every 20 epochs
        if (epoch) % CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
    
    checkpoint.save(file_prefix = checkpoint_prefix)
    
    dt = time.time()-t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  
    
    
# Função para o gerador
def fit_gen(train_ds, first_epoch, epochs, test_ds):
    
    t0 = time.time()
    
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            generate_save_images_gen(generator, example_input, result_folder, filename)
        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            target = input_image
            losses = train_step_gen(input_image, target, epoch)
            
            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every 20 epochs
        if (epoch) % CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
    
    checkpoint.save(file_prefix = checkpoint_prefix)
    
    dt = time.time()-t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  
  
#%% TESTA O CÓDIGO E MOSTRA UMA IMAGEM DO DATASET

inp = load(train_folder+'/male/000016.jpg')
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp/255.0)


#%% PREPARAÇÃO DOS MODELOS

# Criando os modelos
if USE_FULL_GENERATOR: 
    # generator = cyclegan_resnet_generator()
    generator = cyclegan_resnet_adapted_generator()
else:
    encoder = cyclegan_resnet_encoder()
    decoder = cyclegan_resnet_decoder()
    
# disc = discriminator()
disc = patchgan_discriminator()

# Define os otimizadores
if USE_FULL_GENERATOR: 
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
else:
    encoder_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    decoder_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Salva o gerador e o discriminador (principalmente para visualização)
if USE_FULL_GENERATOR: 
    generator.save(model_folder+'ae_generator.h5')
else:
    encoder.save(model_folder+'ae_encoder.h5')
    decoder.save(model_folder+'ae_decoder.h5')
disc.save(model_folder+'ae_discriminator.h5')

#%% EXECUÇÃO

# Prepara os inputs
train_dataset = tf.data.Dataset.list_files(train_folder+'*/*.jpg')
train_size = len(list(train_dataset))
train_dataset = train_dataset.map(load_image_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(test_folder+'*/*.jpg')
test_size = len(list(test_dataset))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


if USE_FULL_GENERATOR: 
    # Prepara o checkpoint (gerador)
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                     discriminator_optimizer = discriminator_optimizer,
                                     generator = generator,
                                     disc = disc)
else:
    # Prepara o checkpoint (encoder/decoder)
    checkpoint = tf.train.Checkpoint(encoder_optimizer = encoder_optimizer,
                                 decoder_optimizer = decoder_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 encoder = encoder,
                                 decoder = decoder,
                                 disc = disc)

# Se for o caso, recupera o checkpoint mais recente
if LOAD_CHECKPOINT:
    print("Carregando checkpoint mais recente...")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint != None:
        checkpoint.restore(latest_checkpoint)
        FIRST_EPOCH = int(latest_checkpoint.split("-")[1]) + 1
    else:
        FIRST_EPOCH = 1

#%% TREINAMENTO

if FIRST_EPOCH <= EPOCHS:
    if USE_FULL_GENERATOR: 
        fit_gen(train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
    else:
        fit(train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
    


#%% Mostra resultado

c = 1
if NUM_TEST_PRINTS > 0:
    for img in test_dataset.take(NUM_TEST_PRINTS):
        filename = "test_results_" + str(c).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
        if USE_FULL_GENERATOR: 
            generate_images(generator, img)
        else:
            generate_images(encoder, decoder, img)
        
        c = c + 1
    

#%% Salva o gerador e o discriminador

if USE_FULL_GENERATOR: 
    generator.save(model_folder+'ae_generator.h5')
else:
    encoder.save(model_folder+'ae_encoder.h5')
    decoder.save(model_folder+'ae_decoder.h5')

disc.save(model_folder+'ae_discriminator.h5')