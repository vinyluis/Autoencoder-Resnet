'''
Prepara o modelo de transfer learning

https://keras.io/guides/transfer_learning/
https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
https://stackoverflow.com/questions/49546922/keras-replacing-input-layer
https://stackoverflow.com/questions/53907681/how-to-fine-tune-a-functional-model-in-keras
'''

# Imports
import os

# Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, load_model

# Módulos próprios
import utils
import networks_general as net

# Modo de inicialização dos pesos
initializer = tf.random_normal_initializer(0., 0.02)

# %% BLOCOS


def upsample_block(x, filters, name_prefix=None, name_suffix=None, norm_type='instancenorm'):
    """Reconstrução da imagem, baseada na Pix2Pix / CycleGAN"""

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = net.PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    if name_prefix is None or name_suffix is None:
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

    else:
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                            kernel_initializer=initializer, use_bias=True,
                                            name=name_prefix + 'upsample_conv2dtrans' + name_suffix)(x)
        x = norm_layer(name=name_prefix + 'upsample_norm' + name_suffix)(x)
        x = tf.keras.layers.ReLU(name=name_prefix + 'upsample_relu' + name_suffix)(x)

    return x


def simple_upsample(x, scale=2, interpolation='bilinear', name_prefix=None, name_suffix=None):
    """Faz um umpsample simplificado, baseado no Progressive Growth of GANs"""

    if name_prefix is None or name_suffix is None:
        x = tf.keras.layers.UpSampling2D(size=(scale, scale), interpolation=interpolation)(x)

    else:
        x = tf.keras.layers.UpSampling2D(size=(scale, scale), interpolation=interpolation,
                                         name=name_prefix + 'upsampling2d' + name_suffix)(x)

    return x


def simple_upsample_block(x, filters, scale=2, kernel_size=(3, 3), interpolation='bilinear', name_prefix=None, name_suffix=None, norm_type='instancenorm'):

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = net.PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    if name_prefix is None or name_suffix is None:

        x = simple_upsample(x, scale=scale, interpolation=interpolation)

        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

    else:
        x = simple_upsample(x, scale=scale, interpolation=interpolation, name_prefix=name_prefix, name_suffix=name_suffix)

        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="same",
                                            kernel_initializer=initializer, use_bias=True, name=name_prefix + 'conv2dtrans1_' + name_suffix)(x)
        x = norm_layer(name=name_prefix + 'upsample_norm1_' + name_suffix)(x)
        x = tf.keras.layers.ReLU(name=name_prefix + 'upsample_relu1_' + name_suffix)(x)

        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="same",
                                            kernel_initializer=initializer, use_bias=True, name=name_prefix + 'conv2dtrans2_' + name_suffix)(x)
        x = norm_layer(name=name_prefix + 'upsample_norm2_' + name_suffix)(x)
        x = tf.keras.layers.ReLU(name=name_prefix + 'upsample_relu2_' + name_suffix)(x)

    return x


# %% BASE DO MODELO DE TRANSFER TRAINING


def get_encoder(generator, encoder_last_layer, trainable):
    '''
    Separa o Encoder
    Para o encoder é fácil, ele usa o mesmo input do modelo, e é só "cortar" ele antes do final
    '''
    encoder = Model(inputs=generator.input, outputs=generator.get_layer(encoder_last_layer).output, name='encoder')
    encoder.trainable = trainable
    return encoder


def get_decoder(generator, encoder_last_layer, decoder_first_layer, trainable):
    '''
    Separa o Decoder
    Se usar o mesmo método do encoder no decoder ele vai dizer que o input não está certo,
    porque ele é output da camada anterior. Deve-se usar um keras.layer.Input
    '''
    # Descobre o tamanho do input e cria um layer para isso
    decoder_input_shape = generator.get_layer(encoder_last_layer).output_shape
    inputlayer = tf.keras.layers.Input(shape=decoder_input_shape[1:])

    # Descobre o índice de cada layer (colocando numa lista)
    layers = []
    for layer in generator.layers:
        layers.append(layer.name)

    # Descobre o índice que eu quero
    layer_index = layers.index(decoder_first_layer)

    # Separa os layers que serão usados
    layers = layers[layer_index:]

    # Cria o modelo
    x = inputlayer
    for layer in layers:
        x = generator.get_layer(layer)(x)
    decoder = Model(inputs=inputlayer, outputs=x, name='decoder')
    decoder.trainable = trainable
    return decoder


def transfer_model(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, generator_path, generator_filename, upsample_type,
                   encoder_last_layer, decoder_first_layer, transfer_trainable, disentanglement=None):
    '''
    Carrega o modelo, separa em encoder e decoder, insere um modelo no meio e retorna o modelo final
    '''
    # Carrega o modelo e separa entre encoder e decoder
    generator = load_model(generator_path + generator_filename)
    encoder = get_encoder(generator, encoder_last_layer, transfer_trainable)
    decoder = get_decoder(generator, encoder_last_layer, decoder_first_layer, transfer_trainable)

    # Cria o modelo final
    inputlayer = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS), name='transfer_model_input')
    x = encoder(inputlayer)
    if upsample_type is not None:
        x = include_vector(IMG_SIZE, NORM_TYPE, upsample_type, disentanglement)(x)
    x = decoder(x)
    transfer_model = Model(inputs=inputlayer, outputs=x)

    return transfer_model


# %% MODELOS DE "MEIO"


def include_vector(IMG_SIZE, NORM_TYPE, upsample_type, disentanglement):
    """Função usada para criar a redução para o vetor latente e depois retornar para a dimensão (31, 31, 256)

    Args:
        IMG_SIZE: dimensões da imagem (IMG_SIZE x IMG_SIZE)
        NORM_TYPE: tipo de normalização
        upsample: 'conv' = Usa convoluções para fazer o upsample pelo bloco "net.upsample"
                  'simple' = Usa o bloco "net.simple_upsample_block"
        disentanglement: Modo de desemaranhamento (de z para w, StyleGAN). Opções = 'none', 'normal' ou 'smooth'
    """

    # Define o tipo de normalização usada
    if NORM_TYPE == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif NORM_TYPE == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif NORM_TYPE == 'pixelnorm':
        norm_layer = net.PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    if not (upsample_type == 'conv' or upsample_type == 'simple'):
        raise utils.TransferUpsampleError(upsample_type)

    inputlayer = tf.keras.layers.Input(shape=(31, 31, 256))
    x = inputlayer

    # Criação do vetor latente
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True, name='middle_conv1')(x)
    x = norm_layer(name='middle_norm1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name='middle_leakyrelu1')(x)

    # Flatten da convolução.
    # Se IMG_size=256, a saída terá 3721 elementos
    # Se IMG_size=128, a saída terá 841 elementos
    x = tf.keras.layers.Flatten(name='middle_flatten')(x)

    if disentanglement == 'smooth':
        # Redução até vecsize (parte do smooth disentanglement)
        if IMG_SIZE == 256:
            x = tf.keras.layers.Dense(units=3072, kernel_initializer=initializer)(x)  # 2048 + 512
            x = tf.keras.layers.Dense(units=2048, kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dense(units=1024, kernel_initializer=initializer)(x)

        elif IMG_SIZE == 128:
            x = tf.keras.layers.Dense(units=768, kernel_initializer=initializer)(x)  # 512 + 256

        # Disentanglement (de z para w, baseado na StyleGAN)
        disentanglement_steps = 5 if IMG_SIZE == 256 else 7
        for i in range(disentanglement_steps):
            x = tf.keras.layers.Dense(units=vecsize, kernel_initializer=initializer)(x)

    elif disentanglement == 'normal':
        # Disentanglement (de z para w, baseado na StyleGAN)
        for i in range(8):
            x = tf.keras.layers.Dense(units=vecsize, kernel_initializer=initializer)(x)

    elif disentanglement is None or disentanglement == 'none':
        x = tf.keras.layers.Dense(units=vecsize, kernel_initializer=initializer)(x)

    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis=1, name='middle_expand_dims1')
    x = tf.expand_dims(x, axis=1, name='middle_expand_dims2')

    # Upsamples
    if upsample_type == 'conv':
        x = upsample_block(x, 512, name_prefix='middle_', name_suffix='1', norm_type=NORM_TYPE)
        x = upsample_block(x, 512, name_prefix='middle_', name_suffix='2', norm_type=NORM_TYPE)
        x = upsample_block(x, 512, name_prefix='middle_', name_suffix='3', norm_type=NORM_TYPE)
        x = upsample_block(x, 512, name_prefix='middle_', name_suffix='4', norm_type=NORM_TYPE)
        x = upsample_block(x, 256, name_prefix='middle_', name_suffix='5', norm_type=NORM_TYPE)

    if upsample_type == 'simple':
        x = simple_upsample_block(x, 512, name_prefix='middle_', name_suffix='1', norm_type=NORM_TYPE)
        x = simple_upsample_block(x, 512, name_prefix='middle_', name_suffix='2', norm_type=NORM_TYPE)
        x = simple_upsample_block(x, 512, name_prefix='middle_', name_suffix='3', norm_type=NORM_TYPE)
        x = simple_upsample_block(x, 512, name_prefix='middle_', name_suffix='4', norm_type=NORM_TYPE)
        x = simple_upsample_block(x, 512, name_prefix='middle_', name_suffix='5', norm_type=NORM_TYPE)

    # Finaliza para deixar com  a dimensão correta (31, 31, 256)
    x = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='valid', name='middle_conv2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name='middle_leakyrelu2')(x)

    # Cria o modelo
    model = Model(inputs=inputlayer, outputs=x, name="bottleneckmodel")

    return model

# %% TESTE


if __name__ == "__main__":

    base_root = ""

    # Faz a configuração do transfer learning, se for selecionado
    transfer_generator_path = base_root + "Experimentos/EXP_R04A_gen_residual_disc_progan/model/"
    transfer_generator_filename = "generator.h5"
    transfer_upsample_type = 'simple'  # 'none', 'simple' ou 'conv'
    transfer_trainable = False
    transfer_encoder_last_layer = 'leaky_re_lu_14'
    transfer_decoder_first_layer = 'conv2d_transpose'

    # Outras configurações e hiperparâmetros
    IMG_SIZE = 128
    OUTPUT_CHANNELS = 3
    NORM_TYPE = 'batchnorm'
    DISENTANGLEMENT = 'smooth'

    # Cria o gerador
    generator = transfer_model(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, transfer_generator_path, transfer_generator_filename,
                               transfer_upsample_type, transfer_encoder_last_layer, transfer_decoder_first_layer,
                               transfer_trainable, DISENTANGLEMENT)

    # Salva o gerador para ver se está certo
    # generator.save("transfer_generator.h5")
