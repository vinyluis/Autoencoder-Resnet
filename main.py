## Main code for Autoencoders
## Created for the Master's degree dissertation
## Vinícius Trevisan 2020-2022

### Imports
import os
import time
from math import ceil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

# Módulos próprios
import networks as net
import utils
import metrics
import transferlearning as transfer

#%% Weights & Biases

import wandb
wandb.init(project='autoencoders', entity='vinyluis', mode="disabled")
# wandb.init(project='autoencoders', entity='vinyluis', mode="online")

#%% Config Tensorflow

# Verifica se a GPU está disponível:
print("---- VERIFICA SE A GPU ESTÁ DISPONÍVEL:")
print(tf.config.list_physical_devices('GPU'))
print("")

#%% HIPERPARÂMETROS E CONFIGURAÇÕES
config = wandb.config # Salva os hiperparametros no Weights & Biases também

# Root do sistema
base_root = ""

# Parâmetros da imagem / dataset
config.IMG_SIZE = 128
config.OUTPUT_CHANNELS = 3
config.USE_CACHE = True
config.DATASET = "CelebaHQ" # "CelebaHQ" ou "InsetosFlickr"

# Parâmetros de treinamento
config.LAMBDA = 1000 # Efeito da Loss L1
config.LAMBDA_DISC = 1 # Ajuste de escala da loss do dicriminador
config.BATCH_SIZE = 8
config.BUFFER_SIZE = 150
config.LEARNING_RATE_G = 1e-5
config.LEARNING_RATE_D = 1e-5
config.EPOCHS = 5
config.LAMBDA_GP = 10 # Intensidade do Gradient Penalty da WGAN-GP
config.NUM_RESIDUAL_BLOCKS = 6 # Número de blocos residuais do gerador CycleGAN
# config.ADAM_BETA_1 e config.FIRST_EPOCH são definidos em código

# Parâmetros das métricas
config.EVALUATE_IS = True
config.EVALUATE_FID = True
config.EVALUATE_L1 = True
config.EVALUATE_PERCENT_OF_DATASET_TRAIN = 0.10
config.EVALUATE_PERCENT_OF_DATASET_VAL = 0.20
config.EVALUATE_PERCENT_OF_DATASET_TEST = 1.00
config.EVALUATE_TRAIN_IMGS = False # Define se vai usar imagens de treino na avaliação
config.EVALUATE_EVERY_EPOCH = True # Define se vai avaliar em cada época ou apenas no final
# METRIC_SAMPLE_SIZE e METRIC_BATCH_SIZE serão definidas em código, para treino e teste

# Configurações de validação
config.VALIDATION = True # Gera imagens da validação
config.EVAL_ITERATIONS = 10 # A cada quantas iterações se faz a avaliação das métricas nas imagens de validação
config.NUM_VAL_PRINTS = 10 # Controla quantas imagens de validação serão feitas. Com -1 plota todo o dataset de validação

# Configurações de teste
config.TEST = True # Teste do modelo
config.NUM_TEST_PRINTS = 500 # Controla quantas imagens de teste serão feitas. Com -1 plota todo o dataset de teste

# Configurações de checkpoint
config.SAVE_CHECKPOINT = True
config.CHECKPOINT_EPOCHS = 1
config.KEEP_CHECKPOINTS = 1
config.LOAD_CHECKPOINT = True
config.SAVE_MODELS = True

# Outras configurações
QUIET_PLOT = True # Controla se as imagens aparecerão na tela, o que impede a execução do código a depender da IDE
SHUTDOWN_AFTER_FINISH = False # Controla se o PC será desligado quando o código terminar corretamente

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp = ""

# Modelo do gerador. Possíveis = 'pix2pix', 'unet', 'cyclegan', 'cyclegan_vetor', 'full_residual', 'full_residual_dis', 'full_residual_smooth',
#                                'simple_decoder', 'simple_decoder_dis', 'simple_decoder_smooth', 'transfer'
config.gen_model = 'unet'

# Modelo do discriminador. Possíveis = 'patchgan', 'progan', 'progan_adapted'
config.disc_model = 'patchgan'

# Tipo de loss. Possíveis = 'patchganloss', 'wgan', 'wgan-gp', 'l1', 'l2'
config.loss_type = 'patchganloss'

# Faz a configuração do transfer learning, se for selecionado
if config.gen_model == 'transfer':
    config.transfer_generator_path = base_root + "Experimentos/EXP11A_gen_resnet_disc_stylegan/model/"
    config.transfer_generator_filename = "ae_generator.h5"
    config.transfer_middle_model = 'simple'
    config.transfer_trainable = False
    config.transfer_encoder_last_layer = 'leaky_re_lu_20'
    config.transfer_decoder_first_layer = 'conv2d_transpose'
    config.transfer_disentangle = True
    config.transfer_smooth_vector = True

# Acerta o flag ADVERSARIAL que indica se o treinamento é adversário (GAN) ou não
if config.loss_type == 'l1' or config.loss_type == 'l2':
    config.ADVERSARIAL = False
else:
    config.ADVERSARIAL = True
    
# Valida se pode ser usado o tipo de loss com o tipo de discriminador
if config.loss_type == 'patchganloss':
    config.ADAM_BETA_1 = 0.5
    if not(config.disc_model == 'patchgan' or config.disc_model == 'patchgan_adapted'
            or  config.disc_model == 'progan_adapted' or  config.disc_model == 'progan'):
        raise utils.LossCompatibilityError(config.loss_type, config.disc_model)
elif config.loss_type == 'wgan' or config.loss_type == 'wgan-gp':
    config.ADAM_BETA_1 = 0.9
else:
    config.ADAM_BETA_1 = 0.9

# Valida o IMG_SIZE
if not(config.IMG_SIZE == 256 or config.IMG_SIZE == 128):
    raise utils.sizeCompatibilityError(config.IMG_SIZE)

# Valida se o número de blocos residuais é válido para o gerador CycleGAN
if config.gen_model == 'cyclegan':
    if not (config.NUM_RESIDUAL_BLOCKS == 6 or config.NUM_RESIDUAL_BLOCKS == 9):
        raise BaseException("O número de blocos residuais do gerador CycleGAN não está correto. Opções = 6 ou 9.")

#%% PREPARA AS PASTAS

### Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Experimentos/'
experiment_folder = experiment_root + 'EXP' + config.exp + '_'
experiment_folder += 'gen_'
experiment_folder += config.gen_model
experiment_folder += '_disc_'
experiment_folder += config.disc_model
experiment_folder += '/'

### Pastas dos resultados
result_folder = experiment_folder + 'results-train/'
result_test_folder = experiment_folder + 'results-test/'
result_val_folder = experiment_folder + 'results-val/'
model_folder = experiment_folder + 'model/'

### Cria as pastas, se não existirem
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
    
### Pasta do checkpoint
checkpoint_dir = experiment_folder + 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

#%% DATASET

### Pastas do dataset
dataset_root = '../../0_Datasets/'

if config.DATASET == 'CelebaHQ':
    dataset_folder = dataset_root + 'celeba_hq/'

elif config.DATASET == 'InsetosFlickr':
    dataset_folder = dataset_root + 'flickr_internetarchivebookimages/'

else:
    raise BaseException("Selecione um dataset válido")

# Pastas de treino, teste e validação
train_folder = dataset_folder + 'train'
test_folder = dataset_folder + 'test'
val_folder = dataset_folder + 'val'

#%% PREPARAÇÃO DOS INPUTS

print("Carregando o dataset...")

# Dataset de treinamento
train_dataset = tf.data.Dataset.list_files(train_folder+'*/*.jpg')
config.TRAIN_SIZE = len(list(train_dataset))
train_dataset = train_dataset.map(lambda x: utils.load_image_train(x, config.IMG_SIZE, 3))
if config.USE_CACHE:
    train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
train_dataset = train_dataset.batch(config.BATCH_SIZE)

# Dataset de teste
test_dataset = tf.data.Dataset.list_files(test_folder+'*/*.jpg')
config.TEST_SIZE = len(list(test_dataset))
test_dataset = test_dataset.map(lambda x: utils.load_image_test(x, config.IMG_SIZE))
if config.USE_CACHE:
    test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(1)

# Dataset de validação
val_dataset = tf.data.Dataset.list_files(val_folder+'*/*.jpg')
config.VAL_SIZE = len(list(val_dataset))
val_dataset = val_dataset.map(lambda x: utils.load_image_test(x, config.IMG_SIZE))
if config.USE_CACHE:
    val_dataset = val_dataset.cache()
val_dataset = val_dataset.batch(1)

print("O dataset de treino tem {} imagens".format(config.TRAIN_SIZE))
print("O dataset de teste tem {} imagens".format(config.TEST_SIZE))
print("O dataset de validação tem {} imagens".format(config.VAL_SIZE))
print("")

#%% DEFINIÇÃO DAS LOSSES

'''
L1: Não há treinamento adversário e o Gerador é treinado apenas com a Loss L1
L2: Idem, com a loss L2
'''
def loss_l1_generator(gen_output, target):
    gan_loss = 0
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # mean absolute error
    total_gen_loss = config.LAMBDA * l1_loss
    return total_gen_loss, gan_loss, l1_loss

def loss_l2_generator(gen_output, target):
    MSE = tf.keras.losses.MeanSquaredError()
    gan_loss = 0
    l2_loss = MSE(target, gen_output) # mean squared error
    # Usando a loss desse jeito, valores entre 0 e 1 serão subestimados. Deve-se tirar a raiz do MSE
    l2_loss = tf.sqrt(l2_loss) # RMSE
    total_gen_loss = config.LAMBDA * l2_loss
    return total_gen_loss, gan_loss, l2_loss

'''
PatchGAN: Em vez de o discriminador usar uma única predição (0 = falsa, 1 = real), o discriminador da PatchGAN (Pix2Pix e CycleGAN) usa
uma matriz 30x30x1, em que cada "pixel" equivale a uma região da imagem, e o discriminador tenta classificar cada região como normal ou falsa
- A Loss do gerador é a Loss de GAN + LAMBDA* L1_Loss em que a Loss de GAN é a BCE entre a matriz 30x30x1 do gerador e uma matriz de mesma
  dimensão preenchida com "1"s, e a L1_Loss é a diferença entre a imagem objetivo e a imagem gerada
- A Loss do discriminador usa apenas a Loss de Gan, mas com uma matriz "0"s para a imagem do gerador (falsa) e uma de "1"s para a imagem real
'''
def loss_patchgan_generator(disc_generated_output, gen_output, target):
    # Lg = GANLoss + LAMBDA * L1_Loss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = BCE(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # mean absolute error
    total_gen_loss = gan_loss + (config.LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def loss_patchgan_discriminator(disc_real_output, disc_generated_output):
    # Ld = RealLoss + FakeLoss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = BCE(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = BCE(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = config.LAMBDA_DISC * (real_loss + fake_loss)
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
def loss_wgan_generator(disc_generated_output, gen_output, target):
    # O output do discriminador é de tamanho BATCH_SIZE x 1, o valor esperado é a média
    gan_loss = -tf.reduce_mean(disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (config.LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def loss_wgan_discriminator(disc_real_output, disc_generated_output):
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
def loss_wgangp_generator(disc_generated_output, gen_output, target):
    return loss_wgan_generator(disc_generated_output, gen_output, target)

def loss_wgangp_discriminator(disc, disc_real_output, disc_generated_output, real_img, generated_img, target):
    fake_loss = tf.reduce_mean(disc_generated_output)
    real_loss = tf.reduce_mean(disc_real_output)
    gp = gradient_penalty_conditional(disc, real_img, generated_img, target)
    total_disc_loss = total_disc_loss = -(real_loss - fake_loss) + config.LAMBDA_GP * gp + (0.001 * tf.reduce_mean(disc_real_output**2))
    return total_disc_loss, real_loss, fake_loss

def gradient_penalty(discriminator, real_img, fake_img, training):
    ''' 
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    From: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb#scrollTo=LhzOUkhYSOPG
    '''
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

def gradient_penalty_conditional(disc, real_img, generated_img, target):
    ''' 
    Adapted to Conditional Discriminators
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    From: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb#scrollTo=LhzOUkhYSOPG
    '''
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


#%% MÉTRICAS DE QUALIDADE

'''
Serão avaliadas IS, FID e L1 de acordo com as flags no início do programa
METRIC_SAMPLE_SIZE e METRIC_BATCH_SIZE serão definidas aqui de acordo com o tamanho
do dataset e o valor em EVALUATE_PERCENT_OF_DATASET
'''

# Configuração dos batches sizes
if config.DATASET == 'CelebaHQ':
    config.METRIC_BATCH_SIZE = 16

elif config.DATASET == 'InsetosFlickr':
    config.METRIC_BATCH_SIZE = 5 # Não há imagens o suficiente para fazer um batch size muito grande
else:
    config.METRIC_BATCH_SIZE = 10

# Configuração dos sample sizes
config.METRIC_SAMPLE_SIZE_TRAIN = int(config.EVALUATE_PERCENT_OF_DATASET_TRAIN * config.TRAIN_SIZE / config.METRIC_BATCH_SIZE)
config.METRIC_SAMPLE_SIZE_TEST = int(config.EVALUATE_PERCENT_OF_DATASET_TEST * config.TEST_SIZE / config.METRIC_BATCH_SIZE)
config.METRIC_SAMPLE_SIZE_VAL = int(config.EVALUATE_PERCENT_OF_DATASET_VAL * config.VAL_SIZE / config.METRIC_BATCH_SIZE)
config.EVALUATED_IMAGES_TRAIN = config.METRIC_SAMPLE_SIZE_TRAIN * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas
config.EVALUATED_IMAGES_TEST = config.METRIC_SAMPLE_SIZE_TEST * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas
config.EVALUATED_IMAGES_VAL = config.METRIC_SAMPLE_SIZE_VAL * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas

#%% FUNÇÕES DE TREINAMENTO

@tf.function
def train_step(generator, discriminator, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        gen_image = generator(input_image, training = True)
    
        disc_real = discriminator([input_image, target], training=True)
        disc_gen = discriminator([gen_image, target], training=True)

        if config.loss_type == 'patchganloss':
            gen_loss, gen_gan_loss, gen_l1_loss = loss_patchgan_generator(disc_gen, gen_image, target)
            disc_loss, disc_real_loss, disc_fake_loss = loss_patchgan_discriminator(disc_real, disc_gen)
    
        elif config.loss_type == 'wgan':
            gen_loss, gen_gan_loss, gen_l1_loss = loss_wgan_generator(disc_gen, gen_image, target)
            disc_loss, disc_real_loss, disc_fake_loss = loss_wgan_discriminator(disc_real, disc_gen)

        elif config.loss_type == 'wgan-gp':
            gen_loss, gen_gan_loss, gen_l1_loss = loss_wgangp_generator(disc_gen, gen_image, target)
            disc_loss, disc_real_loss, disc_fake_loss = loss_wgangp_discriminator(discriminator, disc_real, disc_gen, input_image, gen_image, target)

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
    losses = {
        'gen_total_loss' : gen_loss,
        'gen_gan_loss' : gen_gan_loss,
        'gen_l1_loss' : gen_l1_loss,
        'disc_total_loss': disc_loss,
        'disc_real_loss' : disc_real_loss,
        'disc_fake_loss' : disc_fake_loss,
    }

    return losses

@tf.function
def train_step_not_adversarial(generator, input_image, target):
    with tf.GradientTape() as gen_tape:
        
        gen_image = generator(input_image, training = True)

        if config.loss_type == 'l1':
            gen_loss, gen_gan_loss, gen_l1_loss = loss_l1_generator(gen_image, target)

        elif config.loss_type == 'l2':
            gen_loss, gen_gan_loss, gen_l1_loss = loss_l2_generator(gen_image, target)
            
        # Incluído o else para não dar erro 'gen_loss' is used before assignment
        else:
            gen_loss = 0
            print("Erro de modelo. Selecione uma Loss válida")

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    # Cria um dicionário das losses
    losses = {
        'gen_total_loss' : gen_loss,
        'gen_l1_loss' : gen_l1_loss
    }
    
    return losses

def evaluate_validation_losses(generator, discriminator, input_image, target):
        
    gen_image = generator(input_image, training = True)

    disc_real = discriminator([input_image, target], training=True)
    disc_gen = discriminator([gen_image, target], training=True)
    
    if config.loss_type == 'patchganloss':
        gen_loss, gen_gan_loss, gen_l1_loss = loss_patchgan_generator(disc_gen, gen_image, target)
        disc_loss, disc_real_loss, disc_fake_loss = loss_patchgan_discriminator(disc_real, disc_gen)

    elif config.loss_type == 'wgan':
        gen_loss, gen_gan_loss, gen_l1_loss = loss_wgan_generator(disc_gen, gen_image, target)
        disc_loss, disc_real_loss, disc_fake_loss = loss_wgan_discriminator(disc_real, disc_gen)

    elif config.loss_type == 'wgan-gp':
        gen_loss, gen_gan_loss, gen_l1_loss = loss_wgangp_generator(disc_gen, gen_image, target)
        disc_loss, disc_real_loss, disc_fake_loss = loss_wgangp_discriminator(discriminator, disc_real, disc_gen, input_image, gen_image, target)

    # Incluído o else para não dar erro 'gen_loss' is used before assignment
    else:
        gen_loss = 0
        disc_loss = 0
        print("Erro de modelo. Selecione uma Loss válida")

    # Cria um dicionário das losses
    losses = {
        'gen_total_loss' : gen_loss,
        'gen_gan_loss' : gen_gan_loss,
        'gen_l1_loss' : gen_l1_loss,
        'disc_total_loss': disc_loss,
        'disc_real_loss' : disc_real_loss,
        'disc_fake_loss' : disc_fake_loss,
    }
    
    return losses

def evaluate_validation_losses_not_adversarial(generator, input_image, target):
        
    gen_image = generator(input_image, training = True)

    if config.loss_type == 'l1':
        gen_loss, gen_gan_loss, gen_l1_loss = loss_l1_generator(gen_image, target)

    elif config.loss_type == 'l2':
        gen_loss, gen_gan_loss, gen_l1_loss = loss_l2_generator(gen_image, target)
        
    # Incluído o else para não dar erro 'gen_loss' is used before assignment
    else:
        gen_loss = 0
        print("Erro de modelo. Selecione uma Loss válida")

    # Cria um dicionário das losses
    losses = {
        'gen_total_loss' : gen_loss,
        'gen_l1_loss' : gen_l1_loss
    }
    
    return losses

def fit(generator, discriminator, train_ds, val_ds, first_epoch, epochs, adversarial = True):
    
    print("INICIANDO TREINAMENTO")

    # Verifica se o discriminador existe, caso seja treinamento adversário
    if adversarial == True:
        if discriminator == None:
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

    ########## LOOP DE TREINAMENTO ##########
    for epoch in range(first_epoch, epochs+1):
        t_start = time.time()

        print(utils.get_time_string(), " - Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():

            # Faz o update da Progress Bar
            i = n.numpy() + 1 # Ajuste porque n começa em 0
            progbar.update(i)
            
            # Step de treinamento
            target = input_image

            if adversarial == True:
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
                    losses_val = evaluate_validation_losses(generator, discriminator, example_input, example_input)
                    # Loga as losses de val no weights and biases
                    wandb.log(utils.dict_tensor_to_numpy(losses_val)) 
        
        # Salva o checkpoint
        if config.SAVE_CHECKPOINT:
            if (epoch) % config.CHECKPOINT_EPOCHS == 0:
                ckpt_manager.save()
                print ('\nSalvando checkpoint da época {}'.format(epoch))

        # Gera as imagens após o treinamento desta época
        utils.generate_fixed_images(fixed_train, fixed_val, generator, epoch, epochs, result_folder, QUIET_PLOT)

        # Loga o tempo de duração da época no wandb
        dt = time.time() - t_start
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})

        ### AVALIAÇÃO DAS MÉTRICAS DE QUALIDADE ###
        if (config.EVALUATE_EVERY_EPOCH == True or
            config.EVALUATE_EVERY_EPOCH == False and epoch == epochs):
            print("Avaliação das métricas de qualidade")

            if config.EVALUATE_TRAIN_IMGS:
                # Avaliação para as imagens de treino
                train_sample = train_ds.unbatch().batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TRAIN) # Corrige o tamanho do batch
                metric_results = metrics.evaluate_metrics(train_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
                train_metrics = {k+"_train": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "train" no final das keys
                wandb.log(train_metrics)

            # Avaliação para as imagens de validação
            val_sample = val_ds.unbatch().shuffle(config.BUFFER_SIZE).batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_VAL) # Corrige o tamanho do batch
            metric_results = metrics.evaluate_metrics(val_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
            val_metrics = {k+"_val": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "val" no final das keys
            wandb.log(val_metrics)

        
#%% PREPARAÇÃO DOS MODELOS

# Define se irá ter a restrição de tamanho de peso da WGAN (clipping)
constrained = False
if config.loss_type == 'wgan':
        constrained = True

# ---- GERADORES
if config.gen_model == 'pix2pix':
    generator = net.pix2pix_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS)
elif config.gen_model == 'unet':
    generator = net.unet_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS)
elif config.gen_model == 'cyclegan':
    generator = net.cyclegan_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
elif config.gen_model == 'cyclegan_vetor': 
    generator = net.cyclegan_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, create_latent_vector = True, num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
elif config.gen_model == 'full_residual':
    generator = net.full_residual_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS)
elif config.gen_model == 'full_residual_dis':
    generator = net.full_residual_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, disentanglement = 'normal')
elif config.gen_model == 'full_residual_smooth':
    generator = net.full_residual_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, disentanglement = 'smooth')
elif config.gen_model == 'simple_decoder':
    generator = net.simple_decoder_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS)
elif config.gen_model == 'simple_decoder_dis':
    generator = net.simple_decoder_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, disentanglement = 'normal')
elif config.gen_model == 'simple_decoder_smooth':
    generator = net.simple_decoder_generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, disentanglement = 'smooth')
elif config.gen_model == 'transfer':
    generator = transfer.transfer_model(config.IMG_SIZE, config.OUTPUT_CHANNELS, config.transfer_generator_path, config.transfer_generator_filename, 
    config.transfer_middle_model, config.transfer_encoder_last_layer, config.transfer_decoder_first_layer, config.transfer_trainable,
    config.transfer_disentangle, config.transfer_smooth_vector)
else:
    raise utils.GeneratorError(config.gen_model)

# ---- DISCRIMINADORES
if config.ADVERSARIAL:
    if config.disc_model == 'patchgan':
        disc = net.patchgan_discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, constrained = constrained)
    elif config.disc_model == 'progan_adapted': 
        disc = net.progan_discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, constrained = constrained, output_type = 'patchgan')
    elif config.disc_model == 'progan':
        disc = net.progan_discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, constrained = constrained, output_type = 'unit')
    else:
        raise utils.DiscriminatorError(config.disc_model)
else:
    disc = None

# ---- OTIMIZADORES
generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_G, beta_1=config.ADAM_BETA_1)
if config.ADVERSARIAL:  
    discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_D, beta_1=config.ADAM_BETA_1)

#%% CHECKPOINTS

# Prepara o checkpoint
if config.ADVERSARIAL:
    # Prepara o checkpoint (adversário)
    ckpt = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                               discriminator_optimizer = discriminator_optimizer,
                               generator = generator,
                               disc = disc)
else:
    # Prepara o checkpoint (não adversário)
    ckpt = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                               generator = generator)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep = config.KEEP_CHECKPOINTS)

# Se for o caso, recupera o checkpoint mais recente
if config.LOAD_CHECKPOINT:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint != None:
        print("Carregando checkpoint mais recente...")
        ckpt.restore(latest_checkpoint)
        config.FIRST_EPOCH = int(latest_checkpoint.split("-")[1]) + 1
    else:
        config.FIRST_EPOCH = 1
        

#%% TREINAMENTO

if config.FIRST_EPOCH <= config.EPOCHS:
    fit(generator, disc, train_dataset, val_dataset, config.FIRST_EPOCH, config.EPOCHS, dversarial = config.ADVERSARIAL)


#%% VALIDAÇÃO

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
        filename = "val_results_" + str(c+1).zfill(len(str(num_imgs))) + ".jpg"
        utils.generate_images(generator, image, result_val_folder, filename, QUIET_PLOT = QUIET_PLOT)

#%% TESTE

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
    t1 = time.time()
    for c, image in test_dataset.enumerate():
        # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
        if config.NUM_TEST_PRINTS >= 0 and c >= config.NUM_TEST_PRINTS:
            break

        # Atualização da progbar
        i = c.numpy() + 1
        progbar.update(i)

        # Salva o arquivo
        filename = "test_results_" + str(c+1).zfill(len(str(num_imgs))) + ".jpg"
        utils.generate_images(generator, image, result_test_folder, filename, QUIET_PLOT = QUIET_PLOT)

    dt = time.time() - t1

    # Loga os tempos de inferência no wandb
    if num_imgs != 0:
        mean_inference_time = dt / num_imgs
        wandb.log({'mean inference time (s)': mean_inference_time})

    # Gera métricas do dataset de teste
    print("Iniciando avaliação das métricas de qualidade do dataset de teste")
    test_sample = test_dataset.unbatch().batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TEST) # Corrige o tamanho do batch
    metric_results = metrics.evaluate_metrics(test_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
    test_metrics = {k+"_test": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "_test" no final das keys
    wandb.log(test_metrics)

#%% FINAL

# Finaliza o Weights and Biases
wandb.finish()

## Salva os modelos 
if config.SAVE_MODELS:
    print("Salvando modelos...\n")
    generator.save(model_folder+'generator.h5')
    if config.ADVERSARIAL:
        disc.save(model_folder+'discriminator.h5')

## Desliga o PC ao final do processo, se for o caso
if SHUTDOWN_AFTER_FINISH:
    time.sleep(60)
    os.system("shutdown /s /t 10")