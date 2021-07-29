### Imports
import os
from matplotlib import pyplot as plt

# Módulos próprios
import networks as net
import utils

### Tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

### Funções importantes

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    return image

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

# normalizing the images to [-1, 1]
def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

def load_image_test(image_file):
    input_image = load(image_file)    
    input_image = resize(input_image, IMG_SIZE, IMG_SIZE)
    input_image = normalize(input_image)
    return input_image

### PARÂMETROS
IMG_SIZE = 128

### PASTAS DO DATASET
dataset_folder = 'C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'

train_folder = dataset_folder+'train/'
test_folder = dataset_folder+'val/'


#%% CARREGA OS MODELOS
'''
https://keras.io/guides/transfer_learning/
https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
https://stackoverflow.com/questions/49546922/keras-replacing-input-layer
https://stackoverflow.com/questions/53907681/how-to-fine-tune-a-functional-model-in-keras
'''
from tensorflow.keras.models import Model, load_model

save_folder = "C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/6_GANs/Autoencoder-Resnet/Experimentos/EXP13 - Teste Transfer Learning/"
model_folder = "C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/6_GANs/Autoencoder-Resnet/Experimentos/EXP11B_gen_resnet_disc_patchgan/model/"
model_filename = "ae_generator.h5"

### Carrega o Gerador
trained_generator = load_model(model_folder + model_filename)

### Separa o Encoder
'''
Para o encoder é fácil, ele usa o mesmo input do modelo, e é só "cortar" ele antes do final
'''
trained_encoder = Model(inputs = trained_generator.input, outputs = trained_generator.get_layer("leaky_re_lu_20").output)
trained_encoder.trainable = False
trained_encoder.save(save_folder+'teste_encoder.h5')

### Separa o Decoder
'''
Se usar o mesmo método do encoder no decoder ele vai dizer que o input não está certo, 
porque ele é output da camada anterior. 
'''

# Descobre o tamanho do input e cria um layer para isso
decoder_input_shape = trained_generator.get_layer("leaky_re_lu_20").output_shape
print(decoder_input_shape)
inputlayer = tf.keras.layers.Input(shape=decoder_input_shape[1:])

# Descobre o índice de cada layer (colocando numa lista)
layers = []
for layer in trained_generator.layers:
    layers.append(layer.name)

# Descobre o índice que eu quero
layer_index = layers.index("conv2d_transpose")
print(layer_index)

# Separa os layers que serão usados
layers = layers[layer_index:]

# Cria o modelo
x = inputlayer
for layer in layers:
    x = trained_generator.get_layer(layer)(x)
trained_decoder = Model(inputs = inputlayer, outputs = x)
trained_decoder.trainable = False
trained_decoder.save(save_folder+'teste_decoder.h5')

### Novo modelo completo
complete_model = tf.keras.Sequential()
complete_model.add(trained_encoder)
complete_model.add(trained_decoder)
complete_model.save(save_folder + "complete_model.h5")

#%% TESTA OS MODELOS

test_dataset = tf.data.Dataset.list_files(test_folder+'*/*.jpg')
test_size = len(list(test_dataset))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)

for example_input in test_dataset.take(1):
    utils.generate_save_images_gen(trained_generator, example_input, save_folder, 'teste_generator.jpg')
    utils.generate_save_images(trained_encoder, trained_decoder, example_input, save_folder, 'teste_encdec.jpg')
    utils.generate_save_images_gen(complete_model, example_input, save_folder, 'teste_completo.jpg')