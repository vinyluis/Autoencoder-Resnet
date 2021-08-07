#%% INÍCIO

### Imports
import os
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
from tensorflow.keras.models import load_model

# Módulos próprios
import transferlearning as transfer


#%% HIPERPARÂMETROS

# Parâmetros de validação
BATCH_SIZE = 1
IMG_SIZE = 128
NUM_TEST_PRINTS = 10
QUIET_PLOT = False

# Root do sistema
# base_root = "../"
base_root = ""

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
exp = "17B"

# Código do experimento que será carregado como base do gerador
base_model_exp = "15A"

# Faz a configuração do modelo separado em encoder/decoder
generator_path = base_root + "Experimentos/" + "EXP15A_gen_resnet_disc_stylegan/model/"
generator_filename = "ae_generator.h5"
encoder_last_layer = 'leaky_re_lu_20'
decoder_first_layer = 'conv2d_transpose'

#%% Prepara as pastas

### Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Experimentos/'
experiment_folder = experiment_root + 'EXP' + exp + '_'
experiment_folder += 'BasedOn_'
experiment_folder += base_model_exp
experiment_folder += '/'

### Pastas do dataset
dataset_folder = 'C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'
# dataset_folder = 'C:/Users/Vinícius/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'

train_folder = dataset_folder+'train/'
test_folder = dataset_folder+'val/'

### Pastas dos resultados
result_test_folder = experiment_folder + 'results/'
model_folder = experiment_folder + 'model/'

### Cria as pastas, se não existirem
if not os.path.exists(experiment_root):
    os.mkdir(experiment_root)

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

if not os.path.exists(result_test_folder):
    os.mkdir(result_test_folder)
    
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

#%% FUNÇÕES DE APOIO

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

def interpolate(encoder, decoder, input_img1, input_img2, save_destination = None, filename = None):
    
    # Gera a primeira imagem
    latent1 = encoder(input_img1, training = True)
    img1 = decoder(latent1, training = True)
    
    # Gera a segunda imagem
    latent2 = encoder(input_img2, training = True)
    img2 = decoder(latent2, training = True)

    # Faz as interpolações com 25%, 50% e 75% do latent1 em relação ao latent2
    latent25p = 0.25 * latent1 + 0.75 * latent2
    latent50p = 0.50 * latent1 + 0.50 * latent2
    latent75p = 0.75 * latent1 + 0.25 * latent2

    # Gera as imagens das interpolações
    img25p = decoder(latent25p, training = True)
    img50p = decoder(latent50p, training = True)
    img75p = decoder(latent75p, training = True)

    # Cria o objeto onde serão plotadas as imagens finais
    f = plt.figure(figsize=(30,5))
    
    # print("Latent Vector:")
    # print(latent1)
    
    display_list = [input_img1[0], img1[0], img75p[0], img50p[0], img25p[0], img2[0], input_img2[0]]
    title = ['Imagem 1', '100%', '75%', '50%', '25%', '0%', 'Imagem 2']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    f.show()
    
    if save_destination != None and filename != None:
        f.savefig(save_destination + filename)

    return f


#%% CRIAÇÃO DO MODELO E DATASET

# Carrega o modelo e separa entre encoder e decoder
generator = load_model(generator_path + generator_filename)
encoder = transfer.get_encoder(generator, encoder_last_layer, False)
decoder = transfer.get_decoder(generator, encoder_last_layer, decoder_first_layer, False)
del generator # Libera memória

# Salva os modelos do experimento
encoder.save(model_folder + 'encoder.h5')
decoder.save(model_folder + 'decoder.h5')

# Prepara o dataset
test_dataset = tf.data.Dataset.list_files(test_folder+'*/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.shuffle(NUM_TEST_PRINTS)
test_dataset = test_dataset.batch(BATCH_SIZE)

#%% PLOTA AS INTERPOLAÇÕES

for i in range(NUM_TEST_PRINTS):
    filename = "interpolation_" + str(i+1).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
    examples = test_dataset.take(2)
    example_list = list(examples.as_numpy_iterator())
    img1 = example_list[0]
    img2 = example_list[1]
    interpolate(encoder, decoder, img1, img2, result_test_folder, filename)
