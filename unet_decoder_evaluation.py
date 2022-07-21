"""Funções para teste do U-Net"""

# Imports
import os
from math import ceil
import time

# Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

# Módulos próprios
import utils
import losses
import networks_general as net
import transferlearning as transfer

#%% --- HIPERPARÂMETROS E CONFIGURAÇÕES

# Parâmetros da imagem / dataset
USE_CACHE = True
DATASET = "CelebaHQ"  # "CelebaHQ" ou "InsetosFlickr" ou "CelebaHQ_Small"
USE_RANDOM_JITTER = False

# Parâmetros de rede
NORM_TYPE = "instancenorm"  # "batchnorm", "instancenorm", "pixelnorm"
LAMBDA = 900  # Efeito da Loss L1. Default = 100.
LAMBDA_DISC = 1  # Ajuste de escala da loss do dicriminador
LAMBDA_GP = 10  # Intensidade do Gradient Penalty da WGAN-GP
NUM_RESIDUAL_BLOCKS = 6  # Número de blocos residuais dos geradores residuais
DISENTANGLEMENT = 'smooth'  # 'none', 'normal', 'smooth'

# Parâmetros de treinamento
BATCH_SIZE = 32
BUFFER_SIZE = 100
LEARNING_RATE_G = 1e-5
LEARNING_RATE_D = 1e-5
EPOCHS = 25

# Parâmetros de validação
EVAL_ITERATIONS = 10

# Outras configurações
QUIET_PLOT = True

# Tipo de loss. Possíveis = 'patchganloss', 'wgan', 'wgan-gp', 'l1', 'l2'
loss_type = 'patchganloss'

# Acerta o flag ADVERSARIAL que indica se o treinamento é adversário (GAN) ou não
ADVERSARIAL = True

# Valida se pode ser usado o tipo de loss com o tipo de discriminador
ADAM_BETA_1 = 0.5

#%% --- FUNÇÕES DE APOIO

def unet_decoder(VEC_SHAPE, IMG_SIZE, OUTPUT_CHANNELS, NORM_TYPE):

    '''
    Decoder usado para o teste da U-Net sem conexões de atalho
    '''

    # Modo de inicialização dos pesos
    initializer = tf.random_normal_initializer(0., 0.02)

    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape=VEC_SHAPE)
    x = inputs

    # Upsample (subida)
    # Para cada subida, somar a saída da camada com uma skip connection
    if IMG_SIZE == 256:
        filters_up = [512, 512, 512, 512, 256, 128, 64]
        dropout_up = [True, True, True, False, False, False, False]
    elif IMG_SIZE == 128:
        filters_up = [512, 512, 512, 256, 128, 64]
        dropout_up = [True, True, True, False, False, False]
    for filter, dropout in zip(filters_up, dropout_up):
        x = net.upsample(x, filter, kernel_size=(4, 4), apply_dropout=dropout, norm_type=NORM_TYPE)

    # Última camada
    x = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

#%% --- IMPORTA O MODELO ORIGINAL ---

# Define os paths
trained_unet_path = "Experimentos/EXP_U01A_gen_unet_disc_patchgan/model/"
trained_unet_gen_path = trained_unet_path + "generator.h5"

# Carrega o gerador e separa apenas o encoder
trained_unet_gen = load_model(trained_unet_gen_path)
trained_unet_encoder = transfer.get_encoder(trained_unet_gen, "leaky_re_lu_6", False)

#print("\n\n\n\nOriginal")
#print(trained_unet_gen.weights[0])
#print("\n\n\n\nNovo")
#print(trained_unet_encoder.weights[0])

# Pega os parâmetros de dimensão do vetor latente e da imagem gerada
latent_vector_shape = trained_unet_encoder.output.shape
input_img_shape = trained_unet_encoder.input.shape
IMG_SIZE = input_img_shape[1]
OUTPUT_CHANNELS = input_img_shape[-1]


#%% --- CARREGA OS DATASETS ---
print("Carregando os datasets...")

dataset_root = '../../0_Datasets/'
dataset_folder = dataset_root + 'celeba_hq/'
dataset_filter_string = '*/*/*.jpg'

train_folder = dataset_folder + 'train'
test_folder = dataset_folder + 'test'
val_folder = dataset_folder + 'val'

# Dataset de treinamento
train_dataset = tf.data.Dataset.list_files(train_folder + dataset_filter_string)
TRAIN_SIZE = len(list(train_dataset))
train_dataset = train_dataset.map(lambda x: utils.load_image_train(x, IMG_SIZE, OUTPUT_CHANNELS, USE_RANDOM_JITTER))
if USE_CACHE:
    train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# Dataset de teste
test_dataset = tf.data.Dataset.list_files(test_folder + dataset_filter_string)
TEST_SIZE = len(list(test_dataset))
test_dataset = test_dataset.map(lambda x: utils.load_image_test(x, IMG_SIZE))
if USE_CACHE:
    test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(1)

# Dataset de validação
val_dataset = tf.data.Dataset.list_files(val_folder + dataset_filter_string)
VAL_SIZE = len(list(val_dataset))
val_dataset = val_dataset.map(lambda x: utils.load_image_test(x, IMG_SIZE))
if USE_CACHE:
    val_dataset = val_dataset.cache()
val_dataset = val_dataset.batch(1)

print(f"O dataset de treino tem {TRAIN_SIZE} imagens")
print(f"O dataset de teste tem {TEST_SIZE} imagens")
print(f"O dataset de validação tem {VAL_SIZE} imagens")
print("")



#%% -- Teste

import numpy as np

norms = []
zero_norms = 0
for n, input_image in train_dataset.enumerate():
    c = n.numpy()
    if c%100 == 0 or c == 0:
        print(c)
    latents =  trained_unet_encoder(input_image)
    for latent in latents:
        norm = np.linalg.norm(latent.numpy())
        norms.append(norm)
        if norm == 0:
            zero_norms += 1


print(norms[0])
print(len(norms))
print(zero_norms)





'''

#%% --- USO DO DECODER NÃO TREINADO ---

# -- PREPARA A PASTA DO EXPERIMENTO
base_root = ""
exp = "U02B"

# Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Validação U-Net/'
experiment_folder = experiment_root + 'EXP_' + exp + '_'
experiment_folder += '/'

# Pastas dos resultados
result_folder = experiment_folder + 'results-train/'
result_test_folder = experiment_folder + 'results-test/'
result_val_folder = experiment_folder + 'results-val/'
model_folder = experiment_folder + 'model/'

# Cria as pastas, se não existirem
if not os.path.exists(experiment_root):
    os.mkdir(experiment_root)

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

if not os.path.exists(result_folder):
    os.mkdir(result_folder)

if not os.path.exists(result_test_folder):
    os.mkdir(result_test_folder)

if not os.path.exists(result_val_folder):
    os.mkdir(result_val_folder)

if not os.path.exists(model_folder):
    os.mkdir(model_folder)




# -- CRIA AS REDES

# Cria um novo decoder
new_unet_decoder = unet_decoder(latent_vector_shape[1:], IMG_SIZE, OUTPUT_CHANNELS, 'instancenorm')

# Cria um discriminador
new_unet_discriminator = net.patchgan_discriminator(IMG_SIZE, OUTPUT_CHANNELS)


# -- OTIMIZADORES
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_G, beta_1=ADAM_BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_D, beta_1=ADAM_BETA_1)

# -- FUNÇÕES DE TREINAMENTO

@tf.function
def train_step(generator, discriminator, latent, input_image, target):
    """Realiza um passo de treinamento no framework adversário.

    A função gera a imagem sintética e a discrimina.
    Usando a imagem real e a sintética, são calculadas as losses do gerador e do discriminador.
    Finalmente usa backpropagation para atualizar o gerador e o discriminador, e retorna as losses
    """

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_image = generator(latent, training=True)

        disc_real = discriminator([input_image, target], training=True)
        disc_gen = discriminator([gen_image, target], training=True)

        gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_patchgan_generator(disc_gen, gen_image, target, LAMBDA)
        disc_loss, disc_real_loss, disc_fake_loss = losses.loss_patchgan_discriminator(disc_real, disc_gen, LAMBDA_DISC)


    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # Cria um dicionário das losses
    loss_dict = {
        'gen_total_loss': gen_loss,
        'gen_gan_loss': gen_gan_loss,
        'gen_l1_loss': gen_l1_loss,
        'disc_total_loss': disc_loss,
        'disc_real_loss': disc_real_loss,
        'disc_fake_loss': disc_fake_loss,
    }

    return loss_dict


def evaluate_validation_losses(generator, discriminator, latent, input_image, target):

    """Avalia as losses para imagens de validação no treinamento adversário.

    A função gera a imagem sintética e a discrimina.
    Usando a imagem real e a sintética, são calculadas as losses do gerador e do discriminador.
    Isso é úitil para monitorar o quão bem a rede está generalizando com dados não vistos.
    """

    gen_image = generator(latent, training=True)

    disc_real = discriminator([input_image, target], training=True)
    disc_gen = discriminator([gen_image, target], training=True)

    gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_patchgan_generator(disc_gen, gen_image, target, LAMBDA)
    disc_loss, disc_real_loss, disc_fake_loss = losses.loss_patchgan_discriminator(disc_real, disc_gen, LAMBDA_DISC)

    # Cria um dicionário das losses
    loss_dict = {
        'gen_total_loss_val': gen_loss,
        'gen_gan_loss_val': gen_gan_loss,
        'gen_l1_loss_val': gen_l1_loss,
        'disc_total_loss_val': disc_loss,
        'disc_real_loss_val': disc_real_loss,
        'disc_fake_loss_val': disc_fake_loss,
    }

    return loss_dict


def fit(generator, discriminator, encoder, train_ds, val_ds, first_epoch, epochs):
    """Treina uma rede com um framework adversário (GAN) ou não adversário.

    Esta função treina a rede usando imagens da base de dados de treinamento,
    enquanto mede o desempenho e as losses da rede com a base de validação.

    Inclui a geração de imagens fixas para monitorar a evolução do treinamento por época,
    e o registro de todas as métricas na plataforma Weights and Biases.
    """

    print("INICIANDO TREINAMENTO")

    # Prepara a progression bar
    progbar_iterations = int(ceil(TRAIN_SIZE / BATCH_SIZE))
    progbar = tf.keras.utils.Progbar(progbar_iterations)

    # Separa imagens fixas para acompanhar o treinamento
    for train_input in train_ds.take(1):
        fixed_train = train_input
    for val_input in val_ds.shuffle(1).take(1):
        fixed_val = val_input

    # Mostra como está a geração das imagens antes do treinamento
    # utils.generate_fixed_images(fixed_train, fixed_val, generator, first_epoch - 1, epochs, result_folder, QUIET_PLOT, log_wandb = False)

    # ---------- LOOP DE TREINAMENTO ----------
    for epoch in range(first_epoch, epochs + 1):
        t1 = time.perf_counter()
        print(f"Época: {epoch}")

        # Train
        for n, input_image in train_ds.enumerate():

            # Faz o update da Progress Bar
            i = n.numpy() + 1  # Ajuste porque n começa em 0
            progbar.update(i)

            # Step de treinamento
            target = input_image
            latent =  encoder(input_image)

            print(latent)
            return -1
            
            # Realiza o step de treino adversário
            losses_train = train_step(generator, discriminator, latent, input_image, target)

            # Acrescenta a época, para manter o controle
            losses_train['epoch'] = epoch

            # A cada EVAL_ITERATIONS iterações, avalia as losses para o conjunto de val
            if (n % EVAL_ITERATIONS) == 0 or n == 1 or n == progbar_iterations:
                for example_input in val_dataset.unbatch().batch(BATCH_SIZE).take(1):
                    # Calcula as losses
                    latent = encoder(input_image)
                    losses_val = evaluate_validation_losses(generator, discriminator, latent, example_input, example_input)

        # Gera as imagens após o treinamento desta época
        # utils.generate_fixed_images(fixed_train, fixed_val, generator, epoch, epochs, result_folder, QUIET_PLOT, log_wandb = False)

        # Loga o tempo de duração da época no wandb
        dt = time.perf_counter() - t1
        print(f'Tempo usado para a época {epoch} foi de {dt / 60:.2f} min ({dt:.2f} sec)\n')


# -- Realiza o treinamentos

fit(new_unet_decoder, new_unet_discriminator, trained_unet_encoder, train_dataset, val_dataset, 1, EPOCHS)




'''

