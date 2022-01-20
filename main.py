## Main code for Autoencoders
## Created for the Master's degree dissertation
## Vinícius Trevisan 2020-2022

### Imports
import os
import time
from matplotlib import pyplot as plt
from math import ceil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

# Módulos próprios
import networks as net
import utils
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

# Parâmetros de treinamento
config.LAMBDA = 1000 # Efeito da Loss L1
config.LAMBDA_DISC = 1 # Ajuste de escala da loss do dicriminador
config.BATCH_SIZE = 8
config.BUFFER_SIZE = 150
config.IMG_SIZE = 128
config.LEARNING_RATE_G = 1e-5
config.LEARNING_RATE_D = 1e-5
config.EPOCHS = 5
config.LAMBDA_GP = 10 # Intensidade do Gradient Penalty da WGAN-GP
# config.ADAM_BETA_1 = 0.5 # 0.5 para a PatchGAN e 0.9 para a WGAN - Definido no código
# config.FIRST_EPOCH = 1 # Definido em código, no checkpoint
config.USE_CACHE = True

# Parâmetros de plot
config.QUIET_PLOT = True
config.NUM_TEST_PRINTS = 10

# Controle do Checkpoint
config.CHECKPOINT_EPOCHS = 1
config.LOAD_CHECKPOINT = True
config.KEEP_CHECKPOINTS = 2

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp = ""

# Modelo do gerador. Possíveis = 'unet', 'resnet', 'resnet_vetor', 'full_resnet', 'full_resnet_dis', 'full_resnet_smooth',
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


#%% Prepara as pastas

### Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Experimentos/'
experiment_folder = experiment_root + 'EXP' + config.exp + '_'
experiment_folder += 'gen_'
experiment_folder += config.gen_model
experiment_folder += '_disc_'
experiment_folder += config.disc_model
experiment_folder += '/'

### Pastas do dataset
dataset_root = '../../0_Datasets/celeba_hq/'

train_folder = dataset_root + 'train/'
test_folder = dataset_root + 'val/'

### Pastas dos resultados
result_folder = experiment_folder + 'results-train/'
result_test_folder = experiment_folder + 'results-test/'

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
    
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
### Pasta do checkpoint
checkpoint_dir = experiment_folder + 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

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

#%% FUNÇÕES DO TREINAMENTO

'''
FUNÇÕES DE TREINAMENTO 
'''
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
            disc_loss, disc_real_loss, disc_fake_loss = loss_wgangp_discriminator(disc, disc_real, disc_gen, input_image, gen_image, target)

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

def fit(generator, discriminator, train_ds, first_epoch, epochs, test_ds, adversarial = True):
    
    # Verifica se o discriminador existe, caso seja treinamento adversário
    if adversarial == True:
        if discriminator == None:
            raise BaseException("Erro! Treinamento adversário precisa de um discriminador")

    # Listas para o cálculo da acurácia
    y_real = []
    y_pred = []

    # Prepara a progression bar
    progbar = tf.keras.utils.Progbar(int(ceil(config.TRAIN_SIZE / config.BATCH_SIZE)))

    ########## LOOP DE TREINAMENTO ##########
    for epoch in range(first_epoch, epochs+1):
        t_start = time.time()
        
        # Plota e salva um exemplo
        for example_input in test_ds.take(1):
            filename = "test_epoch_" + str(epoch-1).zfill(len(str(config.EPOCHS))) + ".jpg"
            fig = utils.generate_images(generator, example_input, result_folder, filename)

            # Loga a figura no wandb
            s = "test epoch {}".format(epoch-1)
            wandbfig = wandb.Image(fig, caption="Test Epoch:{}".format(epoch-1))
            wandb.log({s: wandbfig})

            if config.QUIET_PLOT:
                plt.close(fig)

        for example_input in train_ds.take(1):
            filename = "train_epoch_" + str(epoch-1).zfill(len(str(config.EPOCHS))) + ".jpg"
            fig = utils.generate_images(generator, example_input, result_folder, filename)

            # Loga a figura no wandb
            s = "train epoch {}".format(epoch-1)
            wandbfig = wandb.Image(fig, caption="Train Epoch:{}".format(epoch-1))
            wandb.log({s: wandbfig})

            if config.QUIET_PLOT:
                plt.close(fig)

        print(utils.get_time_string(), " - Epoch: ", epoch)
        
        # Train
        i = 0 # Para o progress bar
        for n, input_image in train_ds.enumerate():

            # Faz o update da Progress Bar
            i += 1
            progbar.update(i)
            
            # Step de treinamento
            target = input_image

            if adversarial == True:
                # Realiza o step de treino adversário
                losses = train_step(generator, discriminator, input_image, target)
                # Cálculo da acurácia
                y_real, y_pred, acc = utils.evaluate_accuracy(generator, discriminator, test_ds, y_real, y_pred)
                losses['target_accuracy'] = acc
            else:
                # Realiza o step de treino não adversário
                losses = train_step_not_adversarial(generator, input_image, target)

            # Log as métricas no wandb 
            wandb.log(utils.dict_tensor_to_numpy(losses))   
            
        # saving (checkpoint) the model every x epochs
        if (epoch) % config.CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")

        dt = time.time() - t_start
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})
        
#%% PREPARAÇÃO DOS MODELOS

# Define se irá ter a restrição de tamanho de peso da WGAN (clipping)
constrained = False
if config.loss_type == 'wgan':
        constrained = True

# CRIANDO O MODELO DE GERADOR
if config.gen_model == 'unet':
    generator = net.unet_generator(config.IMG_SIZE)
elif config.gen_model == 'resnet':
    generator = net.resnet_generator(config.IMG_SIZE)
elif config.gen_model == 'resnet_vetor': 
    generator = net.resnet_generator(config.IMG_SIZE, create_latent_vector = True)
elif config.gen_model == 'full_resnet':
    generator = net.full_resnet_generator(config.IMG_SIZE)
elif config.gen_model == 'full_resnet_dis':
    generator = net.full_resnet_generator(config.IMG_SIZE, disentanglement = 'normal')
elif config.gen_model == 'full_resnet_smooth':
    generator = net.full_resnet_generator(config.IMG_SIZE, disentanglement = 'smooth')
elif config.gen_model == 'simple_decoder':
    generator = net.simple_decoder_generator(config.IMG_SIZE)
elif config.gen_model == 'simple_decoder_dis':
    generator = net.simple_decoder_generator(config.IMG_SIZE, disentanglement = 'normal')
elif config.gen_model == 'simple_decoder_smooth':
    generator = net.simple_decoder_generator(config.IMG_SIZE, disentanglement = 'smooth')
elif config.gen_model == 'transfer':
    generator = transfer.transfer_model(config.IMG_SIZE, config.transfer_generator_path, config.transfer_generator_filename, 
    config.transfer_middle_model, config.transfer_encoder_last_layer, config.transfer_decoder_first_layer, config.transfer_trainable,
    config.transfer_disentangle, config.transfer_smooth_vector)
else:
    raise utils.GeneratorError(config.gen_model)

# CRIANDO O MODELO DE DISCRIMINADOR
if config.ADVERSARIAL:
    if config.disc_model == 'patchgan':
        disc = net.patchgan_discriminator(config.IMG_SIZE, constrained = constrained)
    elif config.disc_model == 'progan_adapted': 
        disc = net.progan_discriminator(config.IMG_SIZE, constrained = constrained, output_type = 'patchgan')
    elif config.disc_model == 'progan':
        disc = net.progan_discriminator(config.IMG_SIZE, constrained = constrained, output_type = 'unit')
    else:
        raise utils.DiscriminatorError(config.disc_model)
else:
    disc = None

# Define os otimizadores
generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_G, beta_1=config.ADAM_BETA_1)
if config.ADVERSARIAL:  
    discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE_D, beta_1=config.ADAM_BETA_1)

#%% EXECUÇÃO

# Prepara os inputs
train_dataset = tf.data.Dataset.list_files(train_folder+'*/*.jpg')
config.TRAIN_SIZE = len(list(train_dataset))
train_dataset = train_dataset.map(lambda x: utils.load_image_train(x, config.IMG_SIZE, 3))
if config.USE_CACHE:
    train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
train_dataset = train_dataset.batch(config.BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(test_folder+'*/*.jpg')
config.TEST_SIZE = len(list(test_dataset))
test_dataset = test_dataset.map(lambda x: utils.load_image_test(x, config.IMG_SIZE))
test_dataset = test_dataset.batch(config.BATCH_SIZE)

# Prepara o checkpoint
if config.ADVERSARIAL:
    # Prepara o checkpoint (adversário)
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                    discriminator_optimizer = discriminator_optimizer,
                                    generator = generator,
                                    disc = disc)
else:
    # Prepara o checkpoint (não adversário)
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                    generator = generator)

# Se for o caso, recupera o checkpoint mais recente
if config.LOAD_CHECKPOINT:
    print("Carregando checkpoint mais recente...")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint != None:
        checkpoint.restore(latest_checkpoint)
        config.FIRST_EPOCH = int(latest_checkpoint.split("-")[1]) + 1
    else:
        config.FIRST_EPOCH = 1
        
# Salva o gerador e o discriminador (principalmente para visualização)
generator.save(model_folder+'ae_generator.h5')
if config.ADVERSARIAL:
    disc.save(model_folder+'ae_discriminator.h5')

#%% TREINAMENTO

if config.FIRST_EPOCH <= config.EPOCHS:
    fit(generator, disc, train_dataset, config.FIRST_EPOCH, config.EPOCHS, test_dataset, adversarial = config.ADVERSARIAL)

#%% VALIDAÇÃO

## Após o treinamento, loga uma imagem do dataset de teste para ver como ficou
for example_input in test_dataset.take(1):
    filename = "epoch_" + str(config.EPOCHS).zfill(len(str(config.EPOCHS))) + ".jpg"
    fig = utils.generate_images(generator, example_input, result_folder, filename)

    # Loga a figura no wandb
    s = "epoch {}".format(config.EPOCHS)
    wandbfig = wandb.Image(fig, caption="Epoch:{}".format(config.EPOCHS))
    wandb.log({s: wandbfig})

    if config.QUIET_PLOT:
        plt.close(fig)

## Gera imagens do dataset de teste
c = 1
if config.NUM_TEST_PRINTS > 0:
    for img in test_dataset.take(config.NUM_TEST_PRINTS):
        filename = "test_results_" + str(c).zfill(len(str(config.NUM_TEST_PRINTS))) + ".jpg"
        utils.generate_images(generator, img, result_test_folder, filename)
        c = c + 1
if config.QUIET_PLOT:
    plt.close("all")

## Salva os modelos 
generator.save(model_folder+'ae_generator.h5')
if config.ADVERSARIAL:
    disc.save(model_folder+'ae_discriminator.h5')

wandb.finish()