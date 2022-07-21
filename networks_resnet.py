import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_addons as tfa

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

# %% BLOCOS

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


def residual_downsample_block(input_tensor, filters, norm_type='instancenorm'):

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

    # Convolução da skip connection
    skip = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides = (2,2))(skip)
    skip = norm_layer()(skip)

    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides = (2,2))(x)
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
    x = tf.keras.layers.ReLU()(x)

    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = norm_layer()(x)

    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.ReLU()(x)

    return x


def residual_block_transpose_upsample(input_tensor, filters, norm_type='instancenorm'):

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

    # Convolução da skip connection
    skip = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, 1), strides = (2,2))(skip)
    skip = norm_layer()(skip)

    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3, 3), padding='same', strides = (2,2))(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = norm_layer()(x)

    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.ReLU()(x)

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



# %% GERADORES

def resnet_adapted_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='none'):

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

    # Primeira camada (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3], [3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    if IMG_SIZE == 256:
        # Etapa 256
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)

        # Etapa 128
        x = residual_downsample_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)

    else:
        # Etapa 128
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)

    # Etapas 64 - 16
    for i in range(3):
        x = residual_downsample_block(x, 128, norm_type=NORM_TYPE)
        x = residual_block(x, 128, norm_type=NORM_TYPE)
        x = residual_block(x, 128, norm_type=NORM_TYPE)

    # Etapas 8 - 4
    for i in range(2):
        x = residual_downsample_block(x, 256, norm_type=NORM_TYPE)
        x = residual_block(x, 256, norm_type=NORM_TYPE)
        x = residual_block(x, 256, norm_type=NORM_TYPE)

    # 4 para 1
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same')(x)  # (bs, 512, 4, 4)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=1, kernel_initializer=initializer)(x)  # (bs, 512, 1, 1)
    x = tf.keras.layers.LeakyReLU()(x)
    

    # Criação do vetor latente
    vecsize = 512
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

    # Etapas 2 - 8
    for i in range(3):
        x = residual_block_transpose_upsample(x, 256, norm_type=NORM_TYPE)
        x = residual_block_transpose(x, 256, norm_type=NORM_TYPE)
        x = residual_block_transpose(x, 256, norm_type=NORM_TYPE)

    # Etapas 16 - 64
    for i in range(3):
        x = residual_block_transpose_upsample(x, 128, norm_type=NORM_TYPE)
        x = residual_block_transpose(x, 128, norm_type=NORM_TYPE)
        x = residual_block_transpose(x, 128, norm_type=NORM_TYPE)

    # Etapa 128
    x = residual_block_transpose_upsample(x, 64, norm_type=NORM_TYPE)
    x = residual_block_transpose(x, 64, norm_type=NORM_TYPE)
    x = residual_block_transpose(x, 64, norm_type=NORM_TYPE)

    if IMG_SIZE == 256:
        # Etapa 256
        x = residual_block_transpose_upsample(x, 64, norm_type=NORM_TYPE)
        x = residual_block_transpose(x, 64, norm_type=NORM_TYPE)
        x = residual_block_transpose(x, 64, norm_type=NORM_TYPE)
        
    # Camadas finais
    x = tf.keras.layers.Conv2DTranspose(filters=OUTPUT_CHANNELS, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=initializer, use_bias=True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # Cria o modelo
    return tf.keras.Model(inputs=inputs, outputs=x)
    

# %% DISCRIMINADORES

def resnet_adapted_discriminator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE):

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

    # Inicializa a rede e os inputs
    inp = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])

    # Primeira camada (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3], [3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding="valid", kernel_initializer=initializer, use_bias=True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    if IMG_SIZE == 256:
        # Etapa 256
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)

        # Etapa 128
        x = residual_downsample_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)

    else:
        # Etapa 128
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)
        x = residual_block(x, 64, norm_type=NORM_TYPE)

    # Etapas 64 - 16
    for i in range(3):
        x = residual_downsample_block(x, 128, norm_type=NORM_TYPE)
        x = residual_block(x, 128, norm_type=NORM_TYPE)
        x = residual_block(x, 128, norm_type=NORM_TYPE)

    # Etapas 8 - 4
    for i in range(2):
        x = residual_downsample_block(x, 256, norm_type=NORM_TYPE)
        x = residual_block(x, 256, norm_type=NORM_TYPE)
        x = residual_block(x, 256, norm_type=NORM_TYPE)

    # 4 para 1
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, kernel_initializer=initializer, padding='same')(x)  # (bs, 512, 4, 4)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=1, kernel_initializer=initializer)(x)  # (bs, 512, 1, 1)
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
        print("ResNet Adaptado                            ", resnet_adapted_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE).output.shape)
        print("ResNet Adaptado Disentangled               ", resnet_adapted_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='normal').output.shape)
        print("ResNet Adaptado Smooth Disentangle         ", resnet_adapted_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE, disentanglement='smooth').output.shape)
        print("\nDiscriminadores:")
        print("ResNet Adaptado                            ", resnet_adapted_discriminator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE).output.shape)
        print("")

    """
    gen = resnet_adapted_generator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE)
    gen.save("gen_test.h5")
    disc = resnet_adapted_discriminator(IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE)
    disc.save("disc_test.h5")
    """
