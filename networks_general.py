import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_addons as tfa
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend

# Modo de inicialização dos pesos
initializer = tf.random_normal_initializer(0., 0.02)

# %% CLASSES AUXILIARES


class PixelNormalization(tf.keras.layers.Layer):

    """Pixel Normalization (usada na ProGAN)."""

    def __init__(self):
        super(PixelNormalization, self).__init__()
        self.epsilon = 1e-8

    def call(self, inputs):
        x = inputs
        # axis=-1 -> A normalização atua nos canais
        return x / tf.sqrt(tf.reduce_mean(x**2, axis=-1, keepdims=True) + self.epsilon)


class ClipConstraint(Constraint):

    """
    Clip model weights to a given hypercube
    https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
    """

    # Set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # Clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # Get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# %% BLOCOS

# -- Básicos


def upsample(x, filters, kernel_size=(3, 3), apply_dropout=False, norm_type='instancenorm'):

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding="same",
                                        kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    if apply_dropout:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def downsample(x, filters, kernel_size=(3, 3), apply_norm=True, norm_type='instancenorm'):

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding="same",
                               kernel_initializer=initializer, use_bias=True)(x)
    if apply_norm:
        x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


# -- Residuais


def residual_block(input_tensor, filters, norm_type='instancenorm'):

    '''
    Cria um bloco resnet baseado na Resnet34
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    x = input_tensor
    skip = input_tensor

    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = norm_layer()(x)

    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    return x


def residual_block_transpose(input_tensor, filters, norm_type='instancenorm'):

    '''
    Cria um bloco resnet baseado na Resnet34, mas invertido (convoluções transpostas)
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    x = input_tensor
    skip = input_tensor

    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = norm_layer()(x)

    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def residual_bottleneck_block(input_tensor, filters, norm_type='instancenorm'):

    '''
    Cria um bloco resnet bottleneck, baseado na Resnet50
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    x = input_tensor
    skip = input_tensor

    # Primeira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1))(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Terceira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=(1, 1))(x)
    x = norm_layer()(x)

    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    return x


def residual_downsample_bottleneck_block(input_tensor, filters, norm_type='instancenorm'):

    '''
    Cria um bloco resnet bottleneck, com redução de dimensão, baseado na Resnet50
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    x = input_tensor
    skip = input_tensor

    # Convolução da skip connection
    skip = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=(1, 1))(skip)
    skip = norm_layer()(skip)

    # Primeira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1))(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Terceira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=(1, 1))(x)
    x = norm_layer()(x)

    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    return x


def residual_disc_block(input_tensor, filters, constraint):

    '''
    Cria um bloco resnet baseado na Resnet34, sem normalização, para o discriminador
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''

    x = input_tensor
    skip = input_tensor

    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_constraint=constraint)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_constraint=constraint)(x)

    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    return x


# -- Simple Decoder


def simple_upsample(x, scale=2, interpolation='bilinear'):
    # Faz um umpsample simplificado, baseado no Progressive Growth of GANs
    x = tf.keras.layers.UpSampling2D(size=(scale, scale), interpolation=interpolation)(x)
    return x


def simple_downsample(x, scale=2):
    # Faz um downsample simplificado, baseado no Progressive Growth of GANs
    x = tf.keras.layers.AveragePooling2D(pool_size=(scale, scale))(x)
    return x


def simple_upsample_block(x, filters, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type='instancenorm'):

    # Define o tipo de normalização usada
    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    x = simple_upsample(x, scale=scale, interpolation=interpolation)

    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="same",
                                        kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="same",
                                        kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


# %% GERADORES


def pix2pix_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE):

    # Define os inputs
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])

    # Encoder
    x = inputs
    x = downsample(x, 64, 4, apply_norm=False, norm_type=NORM_TYPE)
    x = downsample(x, 128, 4, norm_type=NORM_TYPE)
    x = downsample(x, 256, 4, norm_type=NORM_TYPE)
    x = downsample(x, 512, 4, norm_type=NORM_TYPE)
    x = downsample(x, 512, 4, norm_type=NORM_TYPE)
    x = downsample(x, 512, 4, norm_type=NORM_TYPE)
    x = downsample(x, 512, 4, norm_type=NORM_TYPE)
    if IMG_SIZE == 256:
        x = downsample(x, 512, 4, norm_type=NORM_TYPE)

    # Decoder
    x = upsample(x, 512, 4, apply_dropout=True, norm_type=NORM_TYPE)
    x = upsample(x, 512, 4, apply_dropout=True, norm_type=NORM_TYPE)
    x = upsample(x, 512, 4, apply_dropout=True, norm_type=NORM_TYPE)
    if IMG_SIZE == 256:
        x = upsample(x, 512, 4, norm_type=NORM_TYPE)
    x = upsample(x, 256, 4, norm_type=NORM_TYPE)
    x = upsample(x, 128, 4, norm_type=NORM_TYPE)
    x = upsample(x, 64, 4, norm_type=NORM_TYPE)

    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def unet_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE):

    '''
    Versão original do gerador U-Net utilizado nos papers Pix2Pix e CycleGAN
    '''

    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    x = inputs

    # Downsample (descida)
    # Para cada descida, salva a saída como uma skip connection
    if IMG_SIZE == 256:
        filters_down = [64, 128, 256, 512, 512, 512, 512, 512]
        norm_down = [False, True, True, True, True, True, True, True]
    elif IMG_SIZE == 128:
        filters_down = [64, 128, 256, 512, 512, 512, 512]
        norm_down = [False, True, True, True, True, True, True]
    skips = []
    for filter, norm in zip(filters_down, norm_down):
        x = downsample(x, filter, kernel_size=(4, 4), apply_norm=norm, norm_type=NORM_TYPE)
        skips.append(x)

    # Upsample (subida)
    # Para cada subida, somar a saída da camada com uma skip connection
    if IMG_SIZE == 256:
        filters_up = [512, 512, 512, 512, 256, 128, 64]
        dropout_up = [True, True, True, False, False, False, False]
    elif IMG_SIZE == 128:
        filters_up = [512, 512, 512, 256, 128, 64]
        dropout_up = [True, True, True, False, False, False]
    skips = skips[-2::-1]  # Inverte as skip connections, e retira a última
    for filter, dropout, skip in zip(filters_up, dropout_up, skips):
        x = upsample(x, filter, kernel_size=(4, 4), apply_dropout=dropout, norm_type=NORM_TYPE)
        x = tf.keras.layers.Concatenate()([x, skip])

    # Última camada
    x = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def residual_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, create_latent_vector=False, num_residual_blocks=6):

    '''
    Adaptado do gerador utilizado nos papers Pix2Pix e CycleGAN
    Modificado de forma a gerar um vetor latente entre o encoder e o decoder
    '''

    # Define o tipo de normalização usada
    if NORM_TYPE == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif NORM_TYPE == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif NORM_TYPE == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    x = inputs

    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3], [3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Blocos Residuais
    for i in range(num_residual_blocks):
        x = residual_block(x, 256, norm_type=NORM_TYPE)

    # Criação do vetor latente
    if create_latent_vector:

        # Criação do vetor latente (alternativa)
        vecsize = 512
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=vecsize, kernel_initializer=initializer)(x)

        # Transforma novamente num tensor de terceira ordem
        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=1)

        # Reconstrução da imagem
        if IMG_SIZE == 256:
            upsamples = 5
        elif IMG_SIZE == 128:
            upsamples = 4

        for i in range(upsamples):
            x = tf.keras.layers.Conv2DTranspose(filters=vecsize, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
            x = norm_layer()(x)
            x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(filters=vecsize / 2, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(filters=vecsize / 4, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(filters=vecsize / 8, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

        # Camadas finais
        x = tf.keras.layers.Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = tf.keras.layers.Activation('tanh')(x)

    else:
        # Reconstrução da imagem
        x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

        # Camadas finais
        x = tf.keras.layers.ZeroPadding2D([[2, 2], [2, 2]])(x)
        x = tf.keras.layers.Conv2D(filters=OUTPUT_CHANNELS, kernel_size=(7, 7), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = tf.keras.layers.Activation('tanh')(x)

    # Cria o modelo
    return tf.keras.Model(inputs=inputs, outputs=x)


# Modelos customizados


def full_residual_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='none', num_residual_blocks=6):

    '''
    Adaptado com base no gerador Resnet da Pix2Pix
    Feito de forma a gerar um vetor latente entre o encoder e o decoder, mas o decoder é também um resnet
    Após o vetor latente, usar 8 camadas Dense para "desembaraçar" o espaço latente, como feito na StyleGAN
    '''

    # Define o tipo de normalização usada
    if NORM_TYPE == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif NORM_TYPE == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif NORM_TYPE == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    x = inputs

    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3], [3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Blocos Residuais
    for i in range(num_residual_blocks):
        x = residual_block(x, 256, norm_type=NORM_TYPE)

    # Criação do vetor latente
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Flatten da convolução
    # Se IMG_SIZE = 256, a saída terá 3721 elementos
    # Se IMG_SIZE = 128, a saída terá 841 elementos
    x = tf.keras.layers.Flatten()(x)

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

    else:
        raise BaseException("Selecione um tipo válido de desemaranhamento")

    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=1)

    # Upsamplings
    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Blocos Residuais
    for i in range(num_residual_blocks):
        x = residual_block_transpose(x, 256, norm_type=NORM_TYPE)

    # Reconstrução pós blocos residuais

    # --
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # --
    if IMG_SIZE == 256:
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, use_bias=True)(x)
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    x = tf.keras.layers.Conv2DTranspose(filters=OUTPUT_CHANNELS, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # Cria o modelo
    return tf.keras.Model(inputs=inputs, outputs=x)


def simple_decoder_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='none', num_residual_blocks=6):

    '''
    Adaptado com base no gerador Resnet da Pix2Pix
    Feito de forma a gerar um vetor latente entre o encoder e o decoder, mas o decoder é também um resnet
    Após o vetor latente, usar 8 camadas Dense para "desembaraçar" o espaço latente, como feito na StyleGAN
    '''

    # Define o tipo de normalização usada
    if NORM_TYPE == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif NORM_TYPE == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    elif NORM_TYPE == 'pixelnorm':
        norm_layer = PixelNormalization
    else:
        raise BaseException("Tipo de normalização desconhecida")

    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    x = inputs

    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3], [3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Blocos Residuais
    for i in range(num_residual_blocks):
        x = residual_block(x, 256, norm_type=NORM_TYPE)

    # Criação do vetor latente
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Flatten da convolução.
    # Se IMG_SIZE = 256, a saída terá 3721 elementos
    # Se IMG_SIZE = 128, a saída terá 841 elementos
    x = tf.keras.layers.Flatten()(x)

    if disentanglement == 'smooth':

        # Redução até vecsize
        if IMG_SIZE == 256:
            x = tf.keras.layers.Dense(units=3072, kernel_initializer=initializer)(x)  # 2048 + 512
            x = tf.keras.layers.Dense(units=2048, kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dense(units=1024, kernel_initializer=initializer)(x)

        elif IMG_SIZE == 128:
            x = tf.keras.layers.Dense(units=768, kernel_initializer=initializer)(x)  # 512 + 256

        # Disentanglement (de z para w, StyleGAN)
        disentanglement_steps = 5 if IMG_SIZE == 256 else 7
        for i in range(disentanglement_steps):
            x = tf.keras.layers.Dense(units=vecsize, kernel_initializer=initializer)(x)

    elif disentanglement == 'normal':

        # Disentanglement (de z para w, baseado na StyleGAN)
        for i in range(8):
            x = tf.keras.layers.Dense(units=vecsize, kernel_initializer=initializer)(x)

    elif disentanglement is None or disentanglement == 'none':
        x = tf.keras.layers.Dense(units=vecsize, kernel_initializer=initializer)(x)

    else:
        raise BaseException("Selecione um tipo válido de desemaranhamento")
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=1)

    # Upsamples
    # Todos os upsamples vão ser feitos com o simple_upsample, seguidos de duas convoluções na mesma dimensão
    if IMG_SIZE == 256:
        x = simple_upsample_block(x, 512, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type=NORM_TYPE)  # 2, 2, 512
    x = simple_upsample_block(x, 512, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type=NORM_TYPE)  # 4, 4, 512 ou 2, 2, 512
    x = simple_upsample_block(x, 512, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type=NORM_TYPE)  # 8, 8, 512 ou 4, 4, 512
    x = simple_upsample_block(x, 512, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type=NORM_TYPE)  # 16, 16, 512 ou 8, 8, 512
    x = simple_upsample_block(x, 256, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type=NORM_TYPE)  # 32, 32, 256 ou 16, 16, 256
    x = simple_upsample_block(x, 128, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type=NORM_TYPE)  # 64, 64, 128 ou 32, 32, 128
    x = simple_upsample_block(x, 64, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type=NORM_TYPE)  # 128, 128, 64 ou 64, 64, 64
    x = simple_upsample_block(x, 32, scale=2, kernel_size=(3, 3), interpolation='bilinear', norm_type=NORM_TYPE)  # 256, 256, 32 ou 128, 128, 32

    # Camadas finais
    x = tf.keras.layers.Conv2DTranspose(filters=OUTPUT_CHANNELS, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # Cria o modelo
    return tf.keras.Model(inputs=inputs, outputs=x)

# %% DISCRIMINADORES


def patchgan_discriminator(IMG_SIZE, OUTPUT_CHANNELS, constrained=False):

    '''
    Versão original do discriminador utilizado nos papers Pix2Pix e CycleGAN
    '''

    # Restrições para o discriminador (usado na WGAN original)
    constraint = ClipConstraint(0.01)
    if constrained is False:
        constraint = None

    # Inicializa a rede e os inputs
    inp = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='target_image')
    # Na implementação em torch, a concatenação ocorre dentro da classe pix2pixmodel
    x = tf.keras.layers.concatenate([inp, tar])

    # Convoluções
    x = tf.keras.layers.Conv2D(64, 4, strides=2, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    if IMG_SIZE == 256:
        x = tf.keras.layers.Conv2D(128, 4, strides=2, kernel_initializer=initializer, padding='valid', kernel_constraint=constraint)(x)
        x = tf.keras.layers.LeakyReLU()(x)
    elif IMG_SIZE == 128:
        x = tf.keras.layers.Conv2D(128, 2, strides=1, kernel_initializer=initializer, padding='valid', kernel_constraint=constraint)(x)
        x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(256, 4, strides=2, kernel_initializer=initializer, padding='valid', kernel_constraint=constraint)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Camada final (30 x 30 x 1) - Para usar o L1 loss
    x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)

    return tf.keras.Model(inputs=[inp, tar], outputs=x)


def progan_discriminator(IMG_SIZE, OUTPUT_CHANNELS, constrained=False, output_type='unit'):

    '''
    Adaptado do discriminador utilizado nos papers ProgGAN e styleGAN
    1ª adaptação é para poder fazer o treinamento supervisionado, mas com a loss adaptada da WGAN ou WGAN-GP
    2ª adaptação é para usar imagens 256x256 (ou IMG_SIZE x IMG_SIZE):
        As primeiras 3 convoluições são mantidas (filters=16, 16, 32) com as dimensões 256 x 256
        Então "pula" para a sexta convolução, que já é originalmente de tamanho 256 x 256 e continua daí para a frente
    '''
    # Restrições para o discriminador (usado na WGAN original)
    constraint = ClipConstraint(0.01)
    if constrained is False:
        constraint = None

    # Inicializa a rede e os inputs
    inp = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])

    # Primeiras três convoluções adaptadas para IMG_SIZE x IMG_SIZE
    x = tf.keras.layers.Conv2D(16, (1, 1), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 16, IMG_SIZE, IMG_SIZE)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 16, IMG_SIZE, IMG_SIZE)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 32, IMG_SIZE, IMG_SIZE)
    x = tf.keras.layers.LeakyReLU()(x)

    if IMG_SIZE == 256:
        # Etapa 256 (convoluções 6 e 7)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 64, 256, 256)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 128, 256, 256)
        x = tf.keras.layers.LeakyReLU()(x)
        x = simple_downsample(x, scale=2)  # (bs, 128, 128, 128)

    # Etapa 128
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 128, 128, 128)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 256, 128, 128)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale=2)  # (bs, 256, 64, 64)

    # Etapa 64
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 256, 64, 64)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 64, 64)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale=2)  # (bs, 512, 32, 32)

    if output_type == 'patchgan':
        # Etapa 32
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 32, 32)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 32, 32)
        x = tf.keras.layers.LeakyReLU()(x)

        # Adaptação para finalizar com 30x30
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer)(x)  # (bs, 30, 30, 1)

    elif output_type == 'unit':
        # Etapa 32
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 32, 32)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 32, 32)
        x = tf.keras.layers.LeakyReLU()(x)
        x = simple_downsample(x, scale=2)  # (bs, 512, 16, 16)

        # Etapa 16
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 16, 16)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 16, 16)
        x = tf.keras.layers.LeakyReLU()(x)
        x = simple_downsample(x, scale=2)  # (bs, 512, 8, 8)

        # Etapa 8
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 8, 8)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 8, 8)
        x = tf.keras.layers.LeakyReLU()(x)
        x = simple_downsample(x, scale=2)  # (bs, 512, 4, 4)

        # Final - 4 para 1
        # Nesse ponto ele faz uma minibatch stddev. Avaliar depois fazer BatchNorm
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same', kernel_constraint=constraint)(x)  # (bs, 512, 4, 4)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(512, (4, 4), strides=1, kernel_initializer=initializer, kernel_constraint=constraint)(x)  # (bs, 512, 1, 1)
        x = tf.keras.layers.LeakyReLU()(x)

        # Finaliza com uma Fully Connected
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1, kernel_constraint=constraint)(x)

    else:
        raise BaseException("Escolha um tipo de saída válida")

    return tf.keras.Model(inputs=[inp, tar], outputs=x)


def residual_discriminator(IMG_SIZE, OUTPUT_CHANNELS, num_residual_blocks=6, constrained=False):

    '''
    Adaptado do GERADOR utilizado nos papers Pix2Pix e CycleGAN
    Modificado para funcionar como um discriminador condicional
    '''

    # Restrições para o discriminador (usado na WGAN original)
    constraint = ClipConstraint(0.01)
    if constrained is False:
        constraint = None

    # Inicializa a rede e os inputs
    inp = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])

    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3], [3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding="valid", kernel_initializer=initializer, kernel_constraint=constraint, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_constraint=constraint, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_constraint=constraint, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Blocos Residuais
    for i in range(num_residual_blocks):
        x = residual_disc_block(x, 256, constraint=constraint)

    # --
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_constraint=constraint, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="valid", kernel_initializer=initializer, kernel_constraint=constraint, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # --
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=initializer, kernel_constraint=constraint, use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Final - 4 para 1
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, kernel_constraint=constraint, padding='same')(x)  # (bs, 512, 4, 4)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=1, kernel_initializer=initializer, kernel_constraint=constraint)(x)  # (bs, 512, 1, 1)
    x = tf.keras.layers.LeakyReLU()(x)

    # Finaliza com uma Fully Connected
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=[inp, tar], outputs=x)


# %% TESTA


#  Só roda quando este arquivo for chamado como main
if __name__ == "__main__":

    # Testa os shapes dos modelos
    OUTPUT_CHANNELS = 3
    NORM_TYPE = 'instancenorm'
    for IMG_SIZE in [256, 128]:
        print(f"\n---- IMG_SIZE = {IMG_SIZE}")
        print("\nGeradores:")
        print("Pix2Pix                                  ", pix2pix_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE).output.shape)
        print("U-Net                                    ", unet_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE).output.shape)
        print("CycleGAN (residual) generator            ", residual_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, create_latent_vector=False).output.shape)
        print("CycleGAN (residual) generator adaptado   ", residual_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, create_latent_vector=True).output.shape)
        print("Full Residual                            ", full_residual_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE).output.shape)
        print("Full Residual Disentangled               ", full_residual_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='normal').output.shape)
        print("Full Residual Smooth Disentangle         ", full_residual_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='smooth').output.shape)
        print("Simple Decoder                           ", simple_decoder_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE).output.shape)
        print("Simple Decoder Disentangled              ", simple_decoder_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='normal').output.shape)
        print("Simple Decoder Smooth Disentangle        ", simple_decoder_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='smooth').output.shape)
        print("\nDiscriminadores:")
        print("PatchGAN                                 ", patchgan_discriminator(IMG_SIZE, OUTPUT_CHANNELS).output.shape)
        print("ProGAN (output_type = unit)              ", progan_discriminator(IMG_SIZE, OUTPUT_CHANNELS, output_type='unit').output.shape)
        print("ProGAN (output_type = patchgan)          ", progan_discriminator(IMG_SIZE, OUTPUT_CHANNELS, output_type='patchgan').output.shape)
        print("Residual                                 ", residual_discriminator(IMG_SIZE, OUTPUT_CHANNELS).output.shape)
        print("")
