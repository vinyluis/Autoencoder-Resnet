""" FUNÇÕES DE APOIO PARA O AUTOENCODER """

import os
import numpy as np
import matplotlib.pyplot as plt
import wandb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf
from tensorflow.keras import backend as K


# %% FUNÇÕES DE APOIO

def dict_tensor_to_numpy(tensor_dict):
    """Transforma tensores guardados em um dicionário em variáveis numéricas.

    Essa função é usada no resultado das losses, que vem no formato de tensor,
    para transforma-las em variáveis numéricas antes de enviar para o Weights
    and Biases.
    """
    numpy_dict = {}
    for k in tensor_dict.keys():
        try:
            numpy_dict[k] = tensor_dict[k].numpy()
        except Exception:
            numpy_dict[k] = tensor_dict[k]
    return numpy_dict


def generate_images(generator, img_input, save_destination=None, filename=None, QUIET_PLOT=True):
    """Usa o gerador para gerar uma imagem sintética a partir de uma imagem de input"""
    img_predict = generator(img_input, training=True)
    f = plt.figure(figsize=(15, 15))

    display_list = [img_input[0], img_predict[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    if save_destination is not None and filename is not None:
        f.savefig(save_destination + filename)

    if not QUIET_PLOT:
        f.show()
        return f
    else:
        plt.close(f)


def generate_fixed_images(fixed_train, fixed_val, generator, epoch, EPOCHS, save_folder, QUIET_PLOT=True, log_wandb=True):

    """Gera a versão sintética das imagens fixas, para acompanhamento.

    Recebe imagens fixas de treinamento e de validação, o gerador, a época atual e o total de épocas.
    Em seguida passa essas imagens pelo gerador para obter a versão sintética delas.
    Finalmente salva as imagens em disco na pasta save_folder e registra as imagens na plataforma Weights and Biases.
    """

    # Train
    filename_train = "train_epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
    fig_train = generate_images(generator, fixed_train, save_folder, filename_train, QUIET_PLOT=False)

    # Val
    filename_val = "val_epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
    fig_val = generate_images(generator, fixed_val, save_folder, filename_val, QUIET_PLOT=False)

    if log_wandb:
        wandb_title = "Época {}".format(epoch)

        wandb_fig_train = wandb.Image(fig_train, caption="Train")
        wandb_title_train = wandb_title + " - Train"

        wandb_fig_val = wandb.Image(fig_val, caption="Val")
        wandb_title_val = wandb_title + " - Val"

        wandb.log({wandb_title_train: wandb_fig_train,
                   wandb_title_val: wandb_fig_val})

    if QUIET_PLOT:
        plt.close(fig_train)
        plt.close(fig_val)


# -- Memória


def print_used_memory(device='GPU:0'):
    mem_info = tf.config.experimental.get_memory_info(device)

    mem_info_current_bytes = mem_info['current']
    mem_info_current_kbytes = mem_info_current_bytes / 1024
    mem_info_current_mbytes = mem_info_current_kbytes / 1024

    mem_info_peak_bytes = mem_info['peak']
    mem_info_peak_kbytes = mem_info_peak_bytes / 1024
    mem_info_peak_mbytes = mem_info_peak_kbytes / 1024

    print(f"Uso de memória: Current = {mem_info_current_mbytes:,.2f} MB, Peak = {mem_info_peak_mbytes:,.2f} MB")
    return {"current_memory_mbytes": mem_info_current_mbytes, "peak_memory_mbytes": mem_info_peak_mbytes}


def get_model_memory_usage(batch_size, model):

    """
    Based on https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    """
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, layer)
        single_layer_mem = 1
        out_shape = layer.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def get_full_dataset_memory_usage(num_imgs, image_size, image_channels, data_type):

    if data_type == tf.float16:
        unit_size = 2.0
    elif data_type == tf.float32:
        unit_size = 4.0
    elif data_type == tf.float64:
        unit_size = 8.0
    else:
        print("Não foi possível obter o data type da imagem")
        unit_size = 1.0

    image_memory_size_bytes = unit_size * (image_size ** 2) * image_channels
    image_memory_size_gbytes = image_memory_size_bytes / (1024.0 ** 3)

    dataset_memory_size_gbytes = num_imgs * image_memory_size_gbytes
    return dataset_memory_size_gbytes


# %% FUNÇÕES DO DATASET

def load(image_file):
    """Função de leitura das imagens."""
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    return image


def normalize(input_image):
    """Normaliza as imagens para o intervalo [-1, 1]"""
    input_image = (input_image / 127.5) - 1
    return input_image


def resize(input_image, height, width):
    """Redimensiona as imagens para width x height"""
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image


def random_crop(input_image, img_size, num_channels):
    """Realiza um corte quadrado aleatório em uma imagem"""
    cropped_image = tf.image.random_crop(value=input_image, size=[img_size, img_size, num_channels])
    return cropped_image


def random_jitter(input_image, img_size, num_channels):
    """Realiza cortes quadrados aleatórios e inverte aleatoriamente uma imagem"""
    # resizing to 286 x 286 x 3
    new_size = int(img_size * 1.117)
    input_image = resize(input_image, new_size, new_size)
    # randomly cropping to IMGSIZE x IMGSIZE x 3
    input_image = random_crop(input_image, img_size, num_channels)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)

    return input_image


def load_image_train(image_file, img_size, num_channels, use_jitter):
    """Carrega uma imagem do dataset de treinamento."""
    input_image = load(image_file)
    if use_jitter:
        input_image = random_jitter(input_image, img_size, num_channels)
    else:
        input_image = resize(input_image, img_size, img_size)
    input_image = normalize(input_image)
    return input_image


def load_image_test(image_file, img_size):
    """Carrega uma imagem do dataset de teste / validação."""
    input_image = load(image_file)
    input_image = resize(input_image, img_size, img_size)
    input_image = normalize(input_image)
    return input_image

# %% TRATAMENTO DE EXCEÇÕES


class GeneratorError(Exception):
    def __init__(self, gen_model):
        print(f"O gerador {gen_model} é desconhecido")


class DiscriminatorError(Exception):
    def __init__(self, disc_model):
        print(f"O discriminador {disc_model} é desconhecido")


class LossError(Exception):
    def __init__(self, loss_type):
        print(f"A loss {loss_type} é desconhecida")


class LossCompatibilityError(Exception):
    def __init__(self, loss_type, disc_model):
        print(f"A loss {loss_type} não é compatível com o discriminador {disc_model}")


class SizeCompatibilityError(Exception):
    def __init__(self, img_size):
        print(f"IMG_SIZE {img_size} não está disponível")


class TransferUpsampleError(Exception):
    def __init__(self, upsample):
        print(f"Tipo de upsampling {upsample} não definido")
