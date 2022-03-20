#!/usr/bin/python3

"""
Main code for Autoencoders
Created for the Master's degree dissertation
Vinícius Trevisan 2020-2022
"""

# --- Imports
import os
import sys
import time
from math import ceil
import traceback

# --- Tensorflow

# Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Verifica se a GPU está disponível:
print("---- VERIFICA SE A GPU ESTÁ DISPONÍVEL:")
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
print("")

# Habilita a alocação de memória dinâmica
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Verifica a versão do Tensorflow
tf_version = tf. __version__
print(f"Utilizando Tensorflow v {tf_version}")
print("")

# --- Módulos próprios
import losses
import utils
import metrics
import networks as net
import transferlearning as transfer

# --- Weights & Biases
import wandb

# wandb.init(project='autoencoders', entity='vinyluis', mode="disabled")
wandb.init(project='autoencoders', entity='vinyluis', mode="online")

# %% HIPERPARÂMETROS E CONFIGURAÇÕES
config = wandb.config  # Salva os hiperparametros no Weights & Biases também

# Salva a versão do python que foi usada no experimento
config.py_version = f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}'

# Salva a versão do TF que foi usada no experimento
config.tf_version = tf_version

# Root do sistema
base_root = ""

# Parâmetros da imagem / dataset
config.IMG_SIZE = 128
config.OUTPUT_CHANNELS = 3
config.USE_CACHE = True
config.DATASET = "CelebaHQ"  # "CelebaHQ" ou "InsetosFlickr" ou "CelebaHQ_Small"
config.USE_RANDOM_JITTER = False

# Parâmetros de rede
config.NORM_TYPE = "instancenorm"  # "batchnorm", "instancenorm", "pixelnorm"
config.LAMBDA = 900  # Efeito da Loss L1. Default = 100.
config.LAMBDA_DISC = 1  # Ajuste de escala da loss do dicriminador
config.LAMBDA_GP = 10  # Intensidade do Gradient Penalty da WGAN-GP
config.NUM_RESIDUAL_BLOCKS = 6  # Número de blocos residuais dos geradores residuais
config.DISENTANGLEMENT = 'smooth'  # 'none', 'normal', 'smooth'
# config.ADAM_BETA_1 e config.FIRST_EPOCH são definidos em código

# Parâmetros de treinamento
config.BATCH_SIZE = 6
config.BUFFER_SIZE = 100
config.LEARNING_RATE_G = 1e-5
config.LEARNING_RATE_D = 1e-5
config.EPOCHS = 25

# Parâmetros das métricas
config.EVALUATE_IS = True
config.EVALUATE_FID = True
config.EVALUATE_L1 = True
config.EVALUATE_PERCENT_OF_DATASET_TRAIN = 0.10
config.EVALUATE_PERCENT_OF_DATASET_VAL = 0.20
config.EVALUATE_PERCENT_OF_DATASET_TEST = 1.00
config.EVALUATE_TRAIN_IMGS = False  # Define se vai usar imagens de treino na avaliação
config.EVALUATE_EVERY_EPOCH = True  # Define se vai avaliar em cada época ou apenas no final
# METRIC_SAMPLE_SIZE e METRIC_BATCH_SIZE serão definidas em código, para treino e teste

# Configurações de validação
config.VALIDATION = True  # Gera imagens da validação
config.EVAL_ITERATIONS = 10  # A cada quantas iterações se faz a avaliação das métricas nas imagens de validação
config.NUM_VAL_PRINTS = 10  # Controla quantas imagens de validação serão feitas. Com -1 plota todo o dataset de validação

# Configurações de teste
config.TEST = True  # Teste do modelo
config.NUM_TEST_PRINTS = -1  # Controla quantas imagens de teste serão feitas. Com -1 plota todo o dataset de teste

# Configurações de checkpoint
config.SAVE_CHECKPOINT = True
config.CHECKPOINT_EPOCHS = 1
config.KEEP_CHECKPOINTS = 1
config.LOAD_CHECKPOINT = False
config.SAVE_MODELS = True

# Outras configurações
QUIET_PLOT = True  # Controla se as imagens aparecerão na tela, o que impede a execução do código a depender da IDE
SHUTDOWN_AFTER_FINISH = False  # Controla se o PC será desligado quando o código terminar corretamente

# %% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp_group = "R11"
config.exp = "R11A"

if config.exp != "":
    print(f"Experimento {config.exp}")

# Modelo do gerador. Possíveis = 'pix2pix', 'unet', 'residual', 'residual_vetor',
#                                'full_residual', 'simple_decoder', 'transfer'
config.gen_model = 'full_residual'

# Modelo do discriminador. Possíveis = 'patchgan', 'progan', 'progan_adapted'
config.disc_model = 'progan'

# Tipo de loss. Possíveis = 'patchganloss', 'wgan', 'wgan-gp', 'l1', 'l2'
config.loss_type = 'wgan-gp'

# Faz a configuração do transfer learning, se for selecionado
if config.gen_model == 'transfer':
    config.transfer_generator_path = base_root + "Experimentos/EXP_R04A_gen_residual_disc_progan/model/"
    config.transfer_generator_filename = "generator.h5"
    config.transfer_upsample_type = 'conv'  # 'none', 'simple' ou 'conv'
    config.transfer_trainable = True
    config.transfer_encoder_last_layer = 'leaky_re_lu_14'
    config.transfer_decoder_first_layer = 'conv2d_transpose'

# Acerta o flag ADVERSARIAL que indica se o treinamento é adversário (GAN) ou não
if config.loss_type == 'l1' or config.loss_type == 'l2':
    config.ADVERSARIAL = False
else:
    config.ADVERSARIAL = True

# Valida se pode ser usado o tipo de loss com o tipo de discriminador
if config.loss_type == 'patchganloss':
    config.ADAM_BETA_1 = 0.5
    if not(config.disc_model == 'patchgan' or config.disc_model == 'progan_adapted' or config.disc_model == 'progan'):
        raise utils.LossCompatibilityError(config.loss_type, config.disc_model)
elif config.loss_type == 'wgan' or config.loss_type == 'wgan-gp':
    config.ADAM_BETA_1 = 0.9
else:
    config.ADAM_BETA_1 = 0.9

# Valida o IMG_SIZE
if not(config.IMG_SIZE == 256 or config.IMG_SIZE == 128):
    raise utils.sizeCompatibilityError(config.IMG_SIZE)

# Valida se o número de blocos residuais é válido para o gerador residual
if not (config.NUM_RESIDUAL_BLOCKS == 6 or config.NUM_RESIDUAL_BLOCKS == 9):
    raise BaseException("O número de blocos residuais do gerador não está correto. Opções = 6 ou 9.")

# Valida se o tipo de normalização é válido
if not (config.NORM_TYPE == 'batchnorm' or config.NORM_TYPE == 'instancenorm' or config.NORM_TYPE == 'pixelnorm'):
    raise BaseException("Tipo de normalização desconhecida.")

# Valida o tipo de disentanglement
if not (config.DISENTANGLEMENT == 'smooth' or config.DISENTANGLEMENT == 'normal'
        or config.DISENTANGLEMENT is None or config.DISENTANGLEMENT == 'none'):
    raise BaseException("Selecione um tipo válido de desemaranhamento")


# %% PREPARA AS PASTAS

# Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Experimentos/'
experiment_folder = experiment_root + 'EXP_' + config.exp + '_'
experiment_folder += 'gen_'
experiment_folder += config.gen_model
if config.ADVERSARIAL:
    experiment_folder += '_disc_'
    experiment_folder += config.disc_model
else:
    experiment_folder += '_notadversarial'
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

# Pasta do checkpoint
checkpoint_dir = experiment_folder + 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# %% DATASET

# Pastas do dataset
dataset_root = '../../0_Datasets/'

if config.DATASET == 'CelebaHQ':
    dataset_folder = dataset_root + 'celeba_hq/'
    dataset_filter_string = '*/*/*.jpg'

elif config.DATASET == 'CelebaHQ_Small':
    dataset_folder = dataset_root + 'celeba_hq_really_small/'
    dataset_filter_string = '*/*/*.jpg'

elif config.DATASET == 'InsetosFlickr':
    dataset_folder = dataset_root + 'flickr_internetarchivebookimages/'
    dataset_filter_string = '*/*.jpg'

else:
    raise BaseException("Selecione um dataset válido")

# Pastas de treino, teste e validação
train_folder = dataset_folder + 'train'
test_folder = dataset_folder + 'test'
val_folder = dataset_folder + 'val'

# %% PREPARAÇÃO DOS INPUTS

print("Carregando o dataset...")

# Dataset de treinamento
train_dataset = tf.data.Dataset.list_files(train_folder + dataset_filter_string)
config.TRAIN_SIZE = len(list(train_dataset))
train_dataset = train_dataset.map(lambda x: utils.load_image_train(x, config.IMG_SIZE, config.OUTPUT_CHANNELS, config.USE_RANDOM_JITTER))
if config.USE_CACHE:
    train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
train_dataset = train_dataset.batch(config.BATCH_SIZE)

# Dataset de teste
test_dataset = tf.data.Dataset.list_files(test_folder + dataset_filter_string)
config.TEST_SIZE = len(list(test_dataset))
test_dataset = test_dataset.map(lambda x: utils.load_image_test(x, config.IMG_SIZE))
if config.USE_CACHE:
    test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(1)

# Dataset de validação
val_dataset = tf.data.Dataset.list_files(val_folder + dataset_filter_string)
config.VAL_SIZE = len(list(val_dataset))
val_dataset = val_dataset.map(lambda x: utils.load_image_test(x, config.IMG_SIZE))
if config.USE_CACHE:
    val_dataset = val_dataset.cache()
val_dataset = val_dataset.batch(1)

print(f"O dataset de treino tem {config.TRAIN_SIZE} imagens")
print(f"O dataset de teste tem {config.TEST_SIZE} imagens")
print(f"O dataset de validação tem {config.VAL_SIZE} imagens")
print("")


# %% MÉTRICAS DE QUALIDADE

'''
Serão avaliadas IS, FID e L1 de acordo com as flags no início do programa
METRIC_SAMPLE_SIZE e METRIC_BATCH_SIZE serão definidas aqui de acordo com o tamanho
do dataset e o valor em EVALUATE_PERCENT_OF_DATASET
'''

# Configuração dos batches sizes
if config.DATASET == 'CelebaHQ':
    config.METRIC_BATCH_SIZE = 32
elif config.DATASET == 'InsetosFlickr':
    config.METRIC_BATCH_SIZE = 5  # Não há imagens o suficiente para fazer um batch size muito grande
elif config.DATASET == 'CelebaHQ_Small':
    config.METRIC_BATCH_SIZE = 16  # Não há imagens o suficiente para fazer um batch size muito grande
else:
    config.METRIC_BATCH_SIZE = 16

# Configuração dos sample sizes
config.METRIC_SAMPLE_SIZE_TRAIN = int(config.EVALUATE_PERCENT_OF_DATASET_TRAIN * config.TRAIN_SIZE / config.METRIC_BATCH_SIZE)
config.METRIC_SAMPLE_SIZE_TEST = int(config.EVALUATE_PERCENT_OF_DATASET_TEST * config.TEST_SIZE / config.METRIC_BATCH_SIZE)
config.METRIC_SAMPLE_SIZE_VAL = int(config.EVALUATE_PERCENT_OF_DATASET_VAL * config.VAL_SIZE / config.METRIC_BATCH_SIZE)
config.EVALUATED_IMAGES_TRAIN = config.METRIC_SAMPLE_SIZE_TRAIN * config.METRIC_BATCH_SIZE  # Apenas para saber quantas imagens serão avaliadas
config.EVALUATED_IMAGES_TEST = config.METRIC_SAMPLE_SIZE_TEST * config.METRIC_BATCH_SIZE  # Apenas para saber quantas imagens serão avaliadas
config.EVALUATED_IMAGES_VAL = config.METRIC_SAMPLE_SIZE_VAL * config.METRIC_BATCH_SIZE  # Apenas para saber quantas imagens serão avaliadas

# %% FUNÇÕES DE TREINAMENTO


@tf.function
def train_step(generator, discriminator, input_image, target):
    """Realiza um passo de treinamento no framework adversário.

    A função gera a imagem sintética e a discrimina.
    Usando a imagem real e a sintética, são calculadas as losses do gerador e do discriminador.
    Finalmente usa backpropagation para atualizar o gerador e o discriminador, e retorna as losses
    """

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_image = generator(input_image, training=True)

        disc_real = discriminator([input_image, target], training=True)
        disc_gen = discriminator([gen_image, target], training=True)

        if config.loss_type == 'patchganloss':
            gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_patchgan_generator(disc_gen, gen_image, target, config.LAMBDA)
            disc_loss, disc_real_loss, disc_fake_loss = losses.loss_patchgan_discriminator(disc_real, disc_gen, config.LAMBDA_DISC)

        elif config.loss_type == 'wgan':
            gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_wgan_generator(disc_gen, gen_image, target, config.LAMBDA)
            disc_loss, disc_real_loss, disc_fake_loss = losses.loss_wgan_discriminator(disc_real, disc_gen)

        elif config.loss_type == 'wgan-gp':
            gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_wgangp_generator(disc_gen, gen_image, target, config.LAMBDA)
            disc_loss, disc_real_loss, disc_fake_loss, gp = losses.loss_wgangp_discriminator(discriminator, disc_real, disc_gen, input_image, gen_image, target, config.LAMBDA_GP)

        # Incluído o else para não dar erro 'gen_loss' is used before assignment
        else:
            gen_loss = 0
            disc_loss = 0
            print("Erro de modelo. Selecione uma Loss válida")

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    # Cria um dicionário das losses
    loss_dict = {
        'gen_total_loss': gen_loss,
        'gen_gan_loss': gen_gan_loss,
        'gen_l1_loss': gen_l1_loss,
        'disc_total_loss': disc_loss,
        'disc_real_loss': disc_real_loss,
        'disc_fake_loss': disc_fake_loss,
    }
    if config.loss_type == 'wgan-gp':
        loss_dict['gp'] = gp

    return loss_dict


@tf.function
def train_step_not_adversarial(generator, input_image, target):
    """Realiza um passo de treinamento no framework não adversário.

    A função gera a imagem sintética e a discrimina.
    Usando a imagem real e a sintética, são calculadas as losses do gerador.
    Finalmente usa backpropagation para atualizar o gerador, e retorna as losses.
    """
    with tf.GradientTape() as gen_tape:

        gen_image = generator(input_image, training=True)

        if config.loss_type == 'l1':
            gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_l1_generator(gen_image, target, config.LAMBDA)

        elif config.loss_type == 'l2':
            gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_l2_generator(gen_image, target, config.LAMBDA)

        # Incluído o else para não dar erro 'gen_loss' is used before assignment
        else:
            gen_loss = 0
            print("Erro de modelo. Selecione uma Loss válida")

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    # Cria um dicionário das losses
    loss_dict = {
        'gen_total_loss': gen_loss,
        'gen_l1_loss': gen_l1_loss
    }

    return loss_dict


def evaluate_validation_losses(generator, discriminator, input_image, target):

    """Avalia as losses para imagens de validação no treinamento adversário.

    A função gera a imagem sintética e a discrimina.
    Usando a imagem real e a sintética, são calculadas as losses do gerador e do discriminador.
    Isso é úitil para monitorar o quão bem a rede está generalizando com dados não vistos.
    """

    gen_image = generator(input_image, training=True)

    disc_real = discriminator([input_image, target], training=True)
    disc_gen = discriminator([gen_image, target], training=True)

    if config.loss_type == 'patchganloss':
        gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_patchgan_generator(disc_gen, gen_image, target, config.LAMBDA)
        disc_loss, disc_real_loss, disc_fake_loss = losses.loss_patchgan_discriminator(disc_real, disc_gen, config.LAMBDA_DISC)

    elif config.loss_type == 'wgan':
        gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_wgan_generator(disc_gen, gen_image, target, config.LAMBDA)
        disc_loss, disc_real_loss, disc_fake_loss = losses.loss_wgan_discriminator(disc_real, disc_gen)

    elif config.loss_type == 'wgan-gp':
        gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_wgangp_generator(disc_gen, gen_image, target, config.LAMBDA)
        disc_loss, disc_real_loss, disc_fake_loss, gp = losses.loss_wgangp_discriminator(discriminator, disc_real, disc_gen, input_image, gen_image, target, config.LAMBDA_GP)

    # Incluído o else para não dar erro 'gen_loss' is used before assignment
    else:
        gen_loss = 0
        disc_loss = 0
        print("Erro de modelo. Selecione uma Loss válida")

    # Cria um dicionário das losses
    loss_dict = {
        'gen_total_loss_val': gen_loss,
        'gen_gan_loss_val': gen_gan_loss,
        'gen_l1_loss_val': gen_l1_loss,
        'disc_total_loss_val': disc_loss,
        'disc_real_loss_val': disc_real_loss,
        'disc_fake_loss_val': disc_fake_loss,
    }
    if config.loss_type == 'wgan-gp':
        loss_dict['gp_val'] = gp

    return loss_dict


def evaluate_validation_losses_not_adversarial(generator, input_image, target):

    """Avalia as losses para imagens de validação, no treinamento não adversário.

    A função gera a imagem sintética e a discrimina.
    Usando a imagem real e a sintética, são calculadas as losses do gerador.
    Isso é úitil para monitorar o quão bem a rede está generalizando com dados não vistos.
    """

    gen_image = generator(input_image, training=True)

    if config.loss_type == 'l1':
        gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_l1_generator(gen_image, target, config.LAMBDA)

    elif config.loss_type == 'l2':
        gen_loss, gen_gan_loss, gen_l1_loss = losses.loss_l2_generator(gen_image, target, config.LAMBDA)

    # Incluído o else para não dar erro 'gen_loss' is used before assignment
    else:
        gen_loss = 0
        print("Erro de modelo. Selecione uma Loss válida")

    # Cria um dicionário das losses
    loss_dict = {
        'gen_total_loss_val': gen_loss,
        'gen_l1_loss_val': gen_l1_loss
    }

    return loss_dict


def fit(generator, discriminator, train_ds, val_ds, first_epoch, epochs, adversarial=True):
    """Treina uma rede com um framework adversário (GAN) ou não adversário.

    Esta função treina a rede usando imagens da base de dados de treinamento,
    enquanto mede o desempenho e as losses da rede com a base de validação.

    Inclui a geração de imagens fixas para monitorar a evolução do treinamento por época,
    e o registro de todas as métricas na plataforma Weights and Biases.
    """

    print("INICIANDO TREINAMENTO")

    # Verifica se o discriminador existe, caso seja treinamento adversário
    if adversarial is True:
        if discriminator is None:
            raise BaseException("Erro! Treinamento adversário precisa de um discriminador")

    # Prepara a progression bar
    progbar_iterations = int(ceil(config.TRAIN_SIZE / config.BATCH_SIZE))
    progbar = tf.keras.utils.Progbar(progbar_iterations)

    # Separa imagens fixas para acompanhar o treinamento
    for train_input in train_ds.take(1):
        fixed_train = train_input
    for val_input in val_ds.shuffle(1).take(1):
        fixed_val = val_input

    # Mostra como está a geração das imagens antes do treinamento
    utils.generate_fixed_images(fixed_train, fixed_val, generator, first_epoch - 1, epochs, result_folder, QUIET_PLOT)

    # Listas para o cálculo da acurácia
    y_real = []
    y_pred = []

    # Uso de memória
    mem_usage = utils.print_used_memory()
    wandb.log(mem_usage)
    print("")

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

            if adversarial:
                # Realiza o step de treino adversário
                losses_train = train_step(generator, discriminator, input_image, target)
                # Cálculo da acurácia com imagens de validação
                y_real, y_pred, acc = metrics.evaluate_accuracy(generator, discriminator, val_ds, y_real, y_pred)
                losses_train['accuracy'] = acc
            else:
                # Realiza o step de treino não adversário
                losses_train = train_step_not_adversarial(generator, input_image, target)

            # Acrescenta a época, para manter o controle
            losses_train['epoch'] = epoch

            # Log as métricas no wandb
            wandb.log(utils.dict_tensor_to_numpy(losses_train))

            # A cada EVAL_ITERATIONS iterações, avalia as losses para o conjunto de val
            if (n % config.EVAL_ITERATIONS) == 0 or n == 1 or n == progbar_iterations:
                for example_input in val_dataset.unbatch().batch(config.BATCH_SIZE).take(1):
                    # Calcula as losses
                    if adversarial:
                        losses_val = evaluate_validation_losses(generator, discriminator, example_input, example_input)
                    else:
                        losses_val = evaluate_validation_losses_not_adversarial(generator, example_input, example_input)

                    # Loga as losses de val no weights and biases
                    wandb.log(utils.dict_tensor_to_numpy(losses_val))

        # Salva o checkpoint
        if config.SAVE_CHECKPOINT:
            if (epoch) % config.CHECKPOINT_EPOCHS == 0:
                ckpt_manager.save()
                print(f'\nSalvando checkpoint da época {epoch}')

        # Gera as imagens após o treinamento desta época
        utils.generate_fixed_images(fixed_train, fixed_val, generator, epoch, epochs, result_folder, QUIET_PLOT)

        # --- AVALIAÇÃO DAS MÉTRICAS DE QUALIDADE ---
        if (config.EVALUATE_EVERY_EPOCH is True or config.EVALUATE_EVERY_EPOCH is False and epoch == epochs):
            print("Avaliando as métricas de qualidade...")

            if config.EVALUATE_TRAIN_IMGS:
                # Avaliação para as imagens de treino
                train_sample = train_ds.unbatch().batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TRAIN)  # Corrige o tamanho do batch
                metric_results = metrics.evaluate_metrics(train_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
                train_metrics = {k + "_train": v for k, v in metric_results.items()}  # Renomeia o dicionário para incluir "train" no final das keys
                wandb.log(train_metrics)

            # Avaliação para as imagens de validação
            val_sample = val_ds.unbatch().shuffle(config.BUFFER_SIZE).batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_VAL)  # Corrige o tamanho do batch
            metric_results = metrics.evaluate_metrics(val_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
            val_metrics = {k + "_val": v for k, v in metric_results.items()}  # Renomeia o dicionário para incluir "val" no final das keys
            wandb.log(val_metrics)

        # Uso de memória
        mem_usage = utils.print_used_memory()
        wandb.log(mem_usage)

        # Loga o tempo de duração da época no wandb
        dt = time.perf_counter() - t1
        print(f'Tempo usado para a época {epoch} foi de {dt / 60:.2f} min ({dt:.2f} sec)\n')
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt / 60})


# %% PREPARAÇÃO DOS MODELOS

# Define se irá ter a restrição de tamanho de peso da WGAN (clipping)
constrained = False
if config.loss_type == 'wgan':
    constrained = True

# ---- GERADORES
if config.gen_model == 'pix2pix':
    generator = net.pix2pix_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, config.NORM_TYPE)
elif config.gen_model == 'unet':
    generator = net.unet_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, config.NORM_TYPE)
elif config.gen_model == 'residual':
    generator = net.residual_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, config.NORM_TYPE, num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
elif config.gen_model == 'residual_vetor':
    generator = net.residual_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, config.NORM_TYPE, create_latent_vector=True, num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
elif config.gen_model == 'full_residual':
    generator = net.full_residual_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, config.NORM_TYPE, disentanglement=config.DISENTANGLEMENT, num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
elif config.gen_model == 'simple_decoder':
    generator = net.simple_decoder_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, config.NORM_TYPE, disentanglement=config.DISENTANGLEMENT, num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
elif config.gen_model == 'transfer':
    generator = transfer.transfer_model(config.IMG_SIZE, config.OUTPUT_CHANNELS, config.NORM_TYPE, config.transfer_generator_path, config.transfer_generator_filename,
                                        config.transfer_upsample_type, config.transfer_encoder_last_layer, config.transfer_decoder_first_layer, config.transfer_trainable,
                                        config.DISENTANGLEMENT)
else:
    raise utils.GeneratorError(config.gen_model)

# ---- DISCRIMINADORES
if config.ADVERSARIAL:
    if config.disc_model == 'patchgan':
        disc = net.patchgan_discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, constrained=constrained)
    elif config.disc_model == 'progan_adapted':
        disc = net.progan_discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, constrained=constrained, output_type='patchgan')
    elif config.disc_model == 'progan':
        disc = net.progan_discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, constrained=constrained, output_type='unit')
    else:
        raise utils.DiscriminatorError(config.disc_model)
else:
    disc = None

# ---- OTIMIZADORES
generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_G, beta_1=config.ADAM_BETA_1)
if config.ADVERSARIAL:
    discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_D, beta_1=config.ADAM_BETA_1)

# %% CONSUMO DE MEMÓRIA
mem_dict = {}

print("Uso de memória dos modelos:")
gen_mem_usage = utils.get_model_memory_usage(config.BATCH_SIZE, generator)
mem_dict['gen_mem_usage_gbytes'] = gen_mem_usage
print(f"Gerador         = {gen_mem_usage:,.2f} GB")
if config.ADVERSARIAL:
    disc_mem_usage = utils.get_model_memory_usage(config.BATCH_SIZE, disc)
    print(f"Discriminador   = {disc_mem_usage:,.2f} GB")
    mem_dict['disc_mem_usage_gbbytes'] = disc_mem_usage

print("Uso de memória dos datasets:")
train_ds_mem_usage = utils.get_full_dataset_memory_usage(config.TRAIN_SIZE, config.IMG_SIZE, config.OUTPUT_CHANNELS, data_type=train_dataset.element_spec.dtype)
test_ds_mem_usage = utils.get_full_dataset_memory_usage(config.TEST_SIZE, config.IMG_SIZE, config.OUTPUT_CHANNELS, data_type=test_dataset.element_spec.dtype)
val_ds_mem_usage = utils.get_full_dataset_memory_usage(config.VAL_SIZE, config.IMG_SIZE, config.OUTPUT_CHANNELS, data_type=val_dataset.element_spec.dtype)
print(f"Train dataset   = {train_ds_mem_usage:,.2f} GB")
print(f"Test dataset    = {test_ds_mem_usage:,.2f} GB")
print(f"Val dataset     = {val_ds_mem_usage:,.2f} GB")
mem_dict['train_ds_mem_usage_gbytes'] = train_ds_mem_usage
mem_dict['test_ds_mem_usage_gbytes'] = test_ds_mem_usage
mem_dict['val_ds_mem_usage_gbytes'] = val_ds_mem_usage
print("")

wandb.log(mem_dict)

# %% CHECKPOINTS

# Prepara o checkpoint
if config.ADVERSARIAL:
    # Prepara o checkpoint (adversário)
    ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer,
                               generator=generator,
                               disc=disc)
else:
    # Prepara o checkpoint (não adversário)
    ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                               generator=generator)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=config.KEEP_CHECKPOINTS)

# Se for o caso, recupera o checkpoint mais recente
if config.LOAD_CHECKPOINT:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        print("Carregando checkpoint mais recente...")
        ckpt.restore(latest_checkpoint)
        config.FIRST_EPOCH = int(latest_checkpoint.split("-")[1]) + 1
    else:
        config.FIRST_EPOCH = 1
else:
    config.FIRST_EPOCH = 1


# %% TREINAMENTO

if config.FIRST_EPOCH <= config.EPOCHS:
    try:
        fit(generator, disc, train_dataset, val_dataset, config.FIRST_EPOCH, config.EPOCHS, adversarial=config.ADVERSARIAL)
    except Exception:
        # Printa  o uso de memória
        mem_usage = utils.print_used_memory()
        wandb.log(mem_usage)
        # Printa o traceback
        traceback.print_exc()
        # Levanta a exceção
        raise BaseException("Erro durante o treinamento")

# %% VALIDAÇÃO

if config.VALIDATION:

    # Gera imagens do dataset de validação
    print("\nCriando imagens do conjunto de validação...")

    # Caso seja -1, plota tudo
    if config.NUM_VAL_PRINTS < 0:
        num_imgs = config.VAL_SIZE
    else:
        num_imgs = config.NUM_VAL_PRINTS

    # Prepara a progression bar
    progbar_iterations = num_imgs
    progbar = tf.keras.utils.Progbar(progbar_iterations)

    # Rotina de plot das imagens de validação
    for c, image in val_dataset.enumerate():
        # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
        if config.NUM_VAL_PRINTS >= 0 and c >= config.NUM_VAL_PRINTS:
            break

        # Atualização da progbar
        i = c.numpy() + 1
        progbar.update(i)

        # Salva o arquivo
        filename = f"val_results_{str(i).zfill(len(str(num_imgs)))}.jpg"
        utils.generate_images(generator, image, result_val_folder, filename, QUIET_PLOT=QUIET_PLOT)

# %% TESTE

if config.TEST:

    # Gera imagens do dataset de teste
    print("\nCriando imagens do conjunto de teste...")

    # Caso seja -1, plota tudo
    if config.NUM_TEST_PRINTS < 0:
        num_imgs = config.TEST_SIZE
    else:
        num_imgs = config.NUM_TEST_PRINTS

    # Prepara a progression bar
    progbar_iterations = num_imgs
    progbar = tf.keras.utils.Progbar(progbar_iterations)

    # Rotina de plot das imagens de teste
    t1 = time.perf_counter()
    for c, image in test_dataset.enumerate():
        # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
        if config.NUM_TEST_PRINTS >= 0 and c >= config.NUM_TEST_PRINTS:
            break

        # Atualização da progbar
        i = c.numpy() + 1
        progbar.update(i)

        # Salva o arquivo
        filename = f"test_results_{str(i).zfill(len(str(num_imgs)))}.jpg"
        utils.generate_images(generator, image, result_test_folder, filename, QUIET_PLOT=QUIET_PLOT)

    dt = time.perf_counter() - t1

    # Loga os tempos de inferência no wandb
    if num_imgs != 0:
        mean_inference_time = dt / num_imgs
        wandb.log({'mean inference time (s)': mean_inference_time})

    # Gera métricas do dataset de teste
    print("Iniciando avaliação das métricas de qualidade do dataset de teste")
    test_sample = test_dataset.unbatch().batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TEST)  # Corrige o tamanho do batch
    metric_results = metrics.evaluate_metrics(test_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
    test_metrics = {k + "_test": v for k, v in metric_results.items()}  # Renomeia o dicionário para incluir "_test" no final das keys
    wandb.log(test_metrics)

# %% FINAL

# Finaliza o Weights and Biases
wandb.finish()

# Salva os modelos
if config.SAVE_MODELS:
    print("Salvando modelos...\n")
    generator.save(model_folder + 'generator.h5')
    if config.ADVERSARIAL:
        disc.save(model_folder + 'discriminator.h5')

# Desliga o PC ao final do processo, se for o caso
if SHUTDOWN_AFTER_FINISH:
    time.sleep(60)
    os.system("shutdown /s /t 10")
