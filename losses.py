## Definition of the losses for the GANs used on this project

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

#%% DEFINIÇÃO DAS LOSSES

'''
L1: Não há treinamento adversário e o Gerador é treinado apenas com a Loss L1
L2: Idem, com a loss L2
'''
@tf.function
def loss_l1_generator(gen_output, target, lambda_l1):
    """Calcula a loss L1 (MAE - distância média absoluta pixel a pixel) entre a imagem gerada e o objetivo."""
    gan_loss = 0
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # mean absolute error
    total_gen_loss = lambda_l1 * l1_loss
    return total_gen_loss, gan_loss, l1_loss

@tf.function
def loss_l2_generator(gen_output, target, lambda_l2):
    """Calcula a loss L2 (RMSE - raiz da distância média quadrada pixel a pixel) entre a imagem gerada e o objetivo."""
    MSE = tf.keras.losses.MeanSquaredError()
    gan_loss = 0
    l2_loss = MSE(target, gen_output) # mean squared error
    # Usando a loss desse jeito, valores entre 0 e 1 serão subestimados. Deve-se tirar a raiz do MSE
    l2_loss = tf.sqrt(l2_loss) # RMSE
    total_gen_loss = lambda_l2 * l2_loss
    return total_gen_loss, gan_loss, l2_loss

'''
PatchGAN: Em vez de o discriminador usar uma única predição (0 = falsa, 1 = real), o discriminador da PatchGAN (Pix2Pix e CycleGAN) usa
uma matriz 30x30x1, em que cada "pixel" equivale a uma região da imagem, e o discriminador tenta classificar cada região como normal ou falsa
- A Loss do gerador é a Loss de GAN + LAMBDA* L1_Loss em que a Loss de GAN é a BCE entre a matriz 30x30x1 do gerador e uma matriz de mesma
  dimensão preenchida com "1"s, e a L1_Loss é a diferença entre a imagem objetivo e a imagem gerada
- A Loss do discriminador usa apenas a Loss de Gan, mas com uma matriz "0"s para a imagem do gerador (falsa) e uma de "1"s para a imagem real
'''
@tf.function
def loss_patchgan_generator(disc_generated_output, gen_output, target, lambda_l1):
    """Calcula a loss de gerador usando BCE no framework Pix2Pix / PatchGAN.

    O gerador quer "enganar" o discriminador, então nesse caso ele é reforçado quando
    a saída do discriminador é 1 (ou uma matriz de 1s) para uma entrada de imagem sintética.
    O BCE (Binary Cross Entropy) avalia o quanto o discriminador acertou ou errou.

    O framework Pix2Pix / PatchGAN inclui também a loss L1 (distância absoluta pixel a pixel) entre a
    imagem gerada e a imagem objetivo (target), para direcionar o aprendizado do gerador.
    """
    # Lg = GANLoss + LAMBDA * L1_Loss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = BCE(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # mean absolute error
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

@tf.function
def loss_patchgan_discriminator(disc_real_output, disc_generated_output, lambda_disc):
    """Calcula a loss dos discriminadores usando BCE.

    Quando a imagem é real, a saída do discriminador deve ser 1 (ou uma matriz de 1s)
    Quando a imagem é sintética, a saída do discriminador deve ser 0 (ou uma matriz de 0s)
    O BCE (Binary Cross Entropy) avalia o quanto o discriminador acertou ou errou.
    """
    # Ld = RealLoss + FakeLoss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = BCE(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = BCE(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = lambda_disc * (real_loss + fake_loss)
    return total_disc_loss, real_loss, fake_loss

'''
Wasserstein GAN (WGAN): A PatchGAN e as GANs clássicas usam a BCE como medida de distância entre a distribuição real e a inferida pelo gerador,
e esse método é na prática a divergência KL. A WGAN usa a distância de Wasserstein, que é mais estável, então evita o mode collapse.
- O discriminador tenta maximizar E(D(x_real)) - E(D(x_fake)), pois quanto maior a distância, pior o gerador está sendo
- O gerador tenta minimizar -E(D(x_fake)), ou seja, o valor esperado (média) da predição do discriminador para a sua imagem
- Os pesos do discriminador precisam passar por Clipping de -0.01 a 0.01 para garantir a continuidade de Lipschitz
Como a WGAN é não-supervisionada, eu vou acrescentar no gerador também a L1 Loss da PatchGAN para comparar com o target,
e usar a WGAN como substituta da GAN Loss

'''
@tf.function
def loss_wgan_generator(disc_generated_output, gen_output, target, lambda_l1):
    """Calcula a loss de wasserstein (WGAN) para o gerador."""
    # O output do discriminador é de tamanho BATCH_SIZE x 1, o valor esperado é a média
    gan_loss = -tf.reduce_mean(disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

@tf.function
def loss_wgan_discriminator(disc_real_output, disc_generated_output):
    """Calcula a loss de wasserstein (WGAN) para o discriminador."""
    # Maximizar E(D(x_real)) - E(D(x_fake)) é equivalente a minimizar -(E(D(x_real)) - E(D(x_fake))) ou E(D(x_fake)) -E(D(x_real))
    fake_loss = tf.reduce_mean(disc_generated_output)
    real_loss = tf.reduce_mean(disc_real_output)
    total_disc_loss = -(real_loss - fake_loss)
    return total_disc_loss, real_loss, fake_loss

'''
Wasserstein GAN - Gradient Penalty (WGAN-GP): A WGAN tem uma forma muito bruta de assegurar a continuidade de Lipschitz, então
os autores criaram o conceito de Gradient Penalty para manter essa condição de uma forma mais suave.
- O gerador tem a MESMA loss da WGAN
- O discriminador, em vez de ter seus pesos limitados pelo clipping, ganha uma penalidade de gradiente que deve ser calculada
'''
def loss_wgangp_generator(disc_generated_output, gen_output, target, lambda_l1):
    """Calcula a loss de wasserstein com gradient-penalty (WGAN-GP) para o gerador."""
    return loss_wgan_generator(disc_generated_output, gen_output, target, lambda_l1)

def loss_wgangp_discriminator(disc, disc_real_output, disc_generated_output, real_img, generated_img, target, lambda_gp):
    """Calcula a loss de wasserstein com gradient-penalty (WGAN-GP) para o discriminador."""
    fake_loss = tf.reduce_mean(disc_generated_output)
    real_loss = tf.reduce_mean(disc_real_output)
    gp = gradient_penalty_conditional(disc, real_img, generated_img, target)
    total_disc_loss = total_disc_loss = -(real_loss - fake_loss) + lambda_gp * gp + (0.001 * tf.reduce_mean(disc_real_output**2))
    return total_disc_loss, real_loss, fake_loss, gp

@tf.function
def gradient_penalty(discriminator, real_img, fake_img, training):
    """Calcula a penalidade de gradiente para a loss de wassertein-gp (WGAN-GP)."""
    # Get the Batch Size
    batch_size = real_img.shape[0]

    # Calcula gamma
    gamma = tf.random.uniform([batch_size, 1, 1, 1])

    # Calcula a imagem interpolada
    interpolated = real_img * gamma + fake_img * (1 - gamma)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)

        # 1. Get the discriminator output for this interpolated image.
        if training == 'direct':
            pred = discriminator(interpolated, training=True) # O discriminador usa duas imagens como entrada
        elif training == 'progressive':
            pred = discriminator(interpolated)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def gradient_penalty_conditional(disc, real_img, generated_img, target):
    """Calcula a penalidade de gradiente para a loss de wassertein-gp (WGAN-GP).
    Adaptada para o uso em discriminadores condicionais.
    """
    # Get the Batch Size
    batch_size = real_img.shape[0]

    # Calcula gamma
    gamma = tf.random.uniform([batch_size, 1, 1, 1])

    # Calcula a imagem interpolada
    interpolated = real_img * gamma + generated_img * (1 - gamma)
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        
        # 1. Get the discriminator output for this interpolated image.
        pred = disc([interpolated, target], training=True) # O discriminador usa duas imagens como entrada

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
