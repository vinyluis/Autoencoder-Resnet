#%% INÍCIO

### Imports
import os
import time
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

# Módulos próprios
import networks as net
import utils
import transferlearning as transfer

#%% Config Tensorflow

# Evita o erro "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize"
# tfconfig = tf.compat.v1.ConfigProto()
# tfconfig.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=tfconfig)

# Verifica se a GPU está disponível:
print("---- VERIFICA SE A GPU ESTÁ DISPONÍVEL:")
print(tf.config.list_physical_devices('GPU'))
# Verifica se a GPU está sendo usada na sessão
# print("---- VERIFICA SE A GPU ESTÁ SENDO USADA NA SESSÃO:")
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# print(sess)
print("")

#%% Weights & Biases

import wandb
# wandb.init(project='autoencoder_resnet', entity='vinyluis', mode="disabled")
wandb.init(project='autoencoder_resnet', entity='vinyluis', mode="online")

#%% HIPERPARÂMETROS
config = wandb.config # Salva os hiperparametros no Weights & Biases também

# Root do sistema
# base_root = "../"
base_root = ""

# Parâmetros de treinamento
config.LAMBDA = 100 # Efeito da Loss L1
config.LAMBDA_DISC = 1 # Ajuste de escala da loss do dicriminador
config.BATCH_SIZE = 1
config.BUFFER_SIZE = 150
config.IMG_SIZE = 128
config.LEARNING_RATE = 1e-5
config.EPOCHS = 3
config.LAMBDA_GP = 10 # Intensidade do Gradient Penalty da WGAN-GP
# config.ADAM_BETA_1 = 0.5 #0.5 para a PatchGAN e 0.9 para a WGAN - Definido no código
# config.FIRST_EPOCH = 1 # Definido em código, no checkpoint

# Parâmetros de plot
config.QUIET_PLOT = True
config.NUM_TEST_PRINTS = 10

# Controle do Checkpoint
config.CHECKPOINT_EPOCHS = 1
config.LOAD_CHECKPOINT = True
config.KEEP_CHECKPOINTS = 2

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp = "14B"

# Modelo do gerador. Possíveis = 'resnet', 'resnet_vetor', 'encoder_decoder', 'full_resnet', 'simple_decoder', 
# 'full_resnet_dis', 'simple_decoder_dis', 'full_resnet_smooth', 'simple_decoder_smooth', 'transfer'
config.gen_model = 'full_resnet'

# Modelo do discriminador. Possíveis = 'patchgan', 'stylegan_adapted', 'stylegan'
config.disc_model = 'stylegan'

# Tipo de loss. Possíveis = 'patchganloss', 'wgan', 'wgan-gp', 'l1'
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

# Acerta o flag USE_FULL_GENERATOR que indica se o gerador é único (full) ou partido (encoder + decoder)
if config.gen_model == 'encoder_decoder':
    config.USE_FULL_GENERATOR = False
else:
    config.USE_FULL_GENERATOR = True

# Acerta o flag ADVERSARIAL que indica se o treinamento é adversário (GAN) ou não
if config.loss_type == 'l1' or config.loss_type == 'l2':
    config.ADVERSARIAL = False
else:
    config.ADVERSARIAL = True
    
# Valida se pode ser usado o tipo de loss com o tipo de discriminador
if config.loss_type == 'patchganloss':
    config.ADAM_BETA_1 = 0.5
    if not(config.disc_model == 'patchgan' or config.disc_model == 'patchgan_adapted'
            or  config.disc_model == 'stylegan_adapted' or  config.disc_model == 'stylegan'):
        raise utils.LossCompatibilityError(config.loss_type, config.disc_model)
if config.loss_type == 'wgan' or config.loss_type == 'wgan-gp':
    config.ADAM_BETA_1 = 0.9
    if not(config.disc_model == 'stylegan'):
        raise utils.LossCompatibilityError(config.loss_type, config.disc_model)

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
#experiment_folder += '_loss_'
#experiment_folder += loss_type
if config.BATCH_SIZE != 1:
    experiment_folder += '_BATCH_' + str(config.BATCH_SIZE)
experiment_folder += '/'

### Pastas do dataset
dataset_folder = 'C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'
# dataset_folder = 'C:/Users/Vinícius/OneDrive/Vinicius/01-Estudos/00_Datasets/celeba_hq/'

train_folder = dataset_folder+'train/'
test_folder = dataset_folder+'val/'

### Pastas dos resultados
result_folder = experiment_folder + 'results-train-autoencoder/'
result_test_folder = experiment_folder + 'results-test-autoencoder/'

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

#%% FUNÇÕES DE APOIO

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    return image

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

def random_crop(input_image):
    cropped_image = tf.image.random_crop(value = input_image, size = [config.IMG_SIZE, config.IMG_SIZE, 3])
    return cropped_image

# normalizing the images to [-1, 1]
def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

# Equivalente a random_jitter = tf.function(random.jitter)
@tf.function()
def random_jitter(input_image):
    # resizing to 286 x 286 x 3
    if config.IMG_SIZE == 256:
        input_image = resize(input_image, 286, 286)
    elif config.IMG_SIZE == 128:
        input_image = resize(input_image, 142, 142)
    
    # randomly cropping to IMGSIZE x IMGSIZE x 3
    input_image = random_crop(input_image)
    
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
    
    return input_image

def load_image_train(image_file):
    input_image = load(image_file)    
    input_image = random_jitter(input_image)
    input_image = normalize(input_image)
    return input_image

def load_image_test(image_file):
    input_image = load(image_file)    
    input_image = resize(input_image, config.IMG_SIZE, config.IMG_SIZE)
    input_image = normalize(input_image)
    return input_image

#%% DEFINIÇÃO DAS LOSSES

'''
L1: Não há treinamento adversário e o Gerador é treinado apenas com a Loss L1
L2: Idem, com a loss L2
'''
def loss_l1_generator(gen_output, target):
    gan_loss = 0
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # mean absolute error
    total_gen_loss = l1_loss
    return total_gen_loss, gan_loss, l1_loss

def loss_l2_generator(gen_output, target):
    gan_loss = 0
    l2_loss = tf.keras.losses.MeanSquaredError(target - gen_output) # mean squared error
    total_gen_loss = l2_loss
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
    total_disc_loss = fake_loss - real_loss
    return total_disc_loss, real_loss, fake_loss

'''
Wasserstein GAN - Gradient Penalty (WGAN-GP): A WGAN tem uma forma muito bruta de assegurar a continuidade de Lipschitz, então
os autores criaram o conceito de Gradient Penalty para manter essa condição de uma forma mais suave.
- O gerador tem a MESMA loss da WGAN
- O discriminador, em vez de ter seus pesos limitados pelo clipping, ganham uma penalidade de gradiente que deve ser calculada
'''
def loss_wgangp_generator(disc_generated_output, gen_output, target):
    return loss_wgan_generator(disc_generated_output, gen_output, target)

def loss_wgangp_discriminator(disc, disc_real_output, disc_generated_output, real_img, generated_img, target):
    total_disc_loss, real_loss, fake_loss = loss_wgan_discriminator(disc_real_output, disc_generated_output)
    gp = gradient_penalty_conditional(disc, real_img, generated_img, target)
    total_disc_loss = total_disc_loss + config.LAMBDA_GP * gp
    return total_disc_loss, real_loss, fake_loss

def gradient_penalty(disc, real_img, generated_img):
    ''' 
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    From: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb#scrollTo=LhzOUkhYSOPG
    '''
    # Get the interpolated image
    alpha = tf.random.normal([config.BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
    diff = generated_img - real_img
    interpolated = real_img + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = disc(interpolated, training=True) # O discriminador usa duas imagens como entrada

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
    # Get the interpolated image
    alpha = tf.random.normal([config.BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
    diff = generated_img - real_img
    interpolated = real_img + alpha * diff
    
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
FUNÇÕES DE TREINAMENTO PARA GERADOR ÚNICO
'''
@tf.function
def train_step(generator, disc, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        gen_image = generator(input_image, training = True)
    
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([gen_image, target], training=True)

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
    
    return (gen_loss, disc_loss, gen_gan_loss, gen_l1_loss, disc_real_loss, disc_fake_loss)

# Função para o gerador
def fit(generator, disc, train_ds, first_epoch, epochs, test_ds):
    
    # Lê arquivo com as losses
    try: 
        loss_df = pd.read_csv(experiment_folder + "losses.csv")
    except:
        loss_df = pd.DataFrame(columns = ["Loss G", "Loss D"])
    
    # Listas para o cálculo da acurácia
    y_real = []
    y_pred = []

    # Inicia o loop de treinamento
    t0 = time.time()
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        # Plota e salva um exemplo
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch-1).zfill(len(str(config.EPOCHS))) + ".jpg"
            fig = utils.generate_save_images_gen(generator, example_input, result_folder, filename)

            # Loga a figura no wandb
            s = "epoch {}".format(epoch-1)
            wandbfig = wandb.Image(fig, caption="Epoch:{}".format(epoch-1))
            wandb.log({s: wandbfig})

            if config.QUIET_PLOT:
                plt.close(fig)

        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            # Step de treinamento
            target = input_image
            gen_loss, disc_loss, gen_gan_loss, gen_l1_loss, disc_real_loss, disc_fake_loss = train_step(generator, disc, input_image, target)

            # Cálculo da acurácia
            y_real, y_pred, acc = utils.evaluate_accuracy(generator, disc, test_ds, y_real, y_pred)

            # Acrescenta a loss no arquivo
            loss_df = loss_df.append({"Loss G": gen_loss.numpy(), "Loss D" : disc_loss.numpy()}, ignore_index = True)
            # Log as métricas no wandb 
            wandb.log({ 'gen_loss': gen_loss.numpy(), 'gen_gan_loss': gen_gan_loss.numpy(), 'gen_l1_loss': gen_l1_loss.numpy(),
                        'disc_loss': disc_loss.numpy(), 'disc_real_loss': disc_real_loss.numpy(), 'disc_fake_loss': disc_fake_loss.numpy(),
                        'test_accuracy': acc})

            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every 20 epochs
        if (epoch) % config.CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
            
        # salva o arquivo de losses a cada época e plota como está ficando
        loss_df.to_csv(experiment_folder + "losses.csv")
        if not config.QUIET_PLOT:
            utils.plot_losses(loss_df)
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})
        
    dt = time.time() - t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  
   

'''
GERADOR SEPARADO EM ENCODER / DECODER
'''
@tf.function
def train_step_encdec(encoder, decoder, disc, input_image, target):
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
        
        latent = encoder(input_image, training = True)
        gen_image = decoder(latent, training = True) 
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)

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

    encoder_gradients = enc_tape.gradient(gen_loss, encoder.trainable_variables)
    decoder_gradients = dec_tape.gradient(gen_loss, decoder.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
    decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_loss, disc_loss, gen_gan_loss, gen_l1_loss, disc_real_loss, disc_fake_loss)

# Função para o encoder/decoder
def fit_encdec(encoder, decoder, disc, train_ds, first_epoch, epochs, test_ds):
    
    # Lê arquivo com as losses
    try: 
        loss_df = pd.read_csv(experiment_folder + "losses.csv")
    except:
        loss_df = pd.DataFrame(columns = ["Loss G", "Loss D"])

    # Listas para o cálculo da acurácia
    y_real = []
    y_pred = []
    
    # Inicia o loop de treinamento
    t0 = time.time()
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        # Plota e salva um exemplo
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch).zfill(len(str(config.EPOCHS))) + ".jpg"
            fig = utils.generate_save_images(encoder, decoder, example_input, result_folder, filename)

            # Loga a figura no wandb
            s = "epoch {}".format(epoch -1)
            wandbfig = wandb.Image(fig, caption="Epoch:{}".format(epoch-1))
            wandb.log({s: wandbfig})

            if config.QUIET_PLOT:
                plt.close(fig)

        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            # Step de treinamento
            target = input_image
            gen_loss, disc_loss, gen_gan_loss, gen_l1_loss, disc_real_loss, disc_fake_loss = train_step_encdec(encoder, decoder, disc, input_image, target)
            
            # Cálculo da acurácia
            y_real, y_pred, acc = utils.evaluate_accuracy(generator, disc, test_ds, y_real, y_pred)

            # Acrescenta a loss no arquivo
            loss_df = loss_df.append({"Loss G": gen_loss.numpy(), "Loss D" : disc_loss.numpy()}, ignore_index = True)
            # Log as métricas no wandb 
            wandb.log({ 'gen_loss': gen_loss.numpy(), 'gen_gan_loss': gen_gan_loss.numpy(), 'gen_l1_loss': gen_l1_loss.numpy(),
                        'disc_loss': disc_loss.numpy(), 'disc_real_loss': disc_real_loss.numpy(), 'disc_fake_loss': disc_fake_loss.numpy(),
                        'test_accuracy': acc})

            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every x epochs
        if (epoch) % config.CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
            
        # salva o arquivo de losses a cada época e plota como está ficando
        loss_df.to_csv(experiment_folder + "losses.csv")
        if not config.QUIET_PLOT:
            utils.plot_losses(loss_df)
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})
    
    dt = time.time()-t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  

'''
FUNÇÕES DE TREINAMENTO SEM DISCRIMINADOR (ADVERSARIAL = FALSE)
'''
@tf.function
def train_step_nodisc(generator, input_image, target):
    with tf.GradientTape() as gen_tape:
        
        gen_image = generator(input_image, training = True)
    
        # disc_real = disc([input_image, target], training=True)
        # disc_gen = disc([gen_image, target], training=True)

        if config.loss_type == 'l1':
            gen_loss, gen_gan_loss, gen_l1_loss = loss_l1_generator(gen_image, target)
            disc_loss = 0
            disc_real_loss = 0
            disc_fake_loss = 0

        if config.loss_type == 'l2':
            gen_loss, gen_gan_loss, gen_l1_loss = loss_l2_generator(gen_image, target)
            disc_loss = 0
            disc_real_loss = 0
            disc_fake_loss = 0
            
        # Incluído o else para não dar erro 'gen_loss' is used before assignment
        else:
            gen_loss = 0
            disc_loss = 0
            print("Erro de modelo. Selecione uma Loss válida")

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    
    return (gen_loss, disc_loss, gen_gan_loss, gen_l1_loss, disc_real_loss, disc_fake_loss)


def fit_nodisc(generator, train_ds, first_epoch, epochs, test_ds):
    
    # Lê arquivo com as losses
    try: 
        loss_df = pd.read_csv(experiment_folder + "losses.csv")
    except:
        loss_df = pd.DataFrame(columns = ["Loss G", "Loss D"])

    # Listas para o cálculo da acurácia
    y_real = []
    y_pred = []
    
    # Inicia o loop de treinamento
    t0 = time.time()
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        # Plota e salva um exemplo
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch-1).zfill(len(str(config.EPOCHS))) + ".jpg"
            fig = utils.generate_save_images_gen(generator, example_input, result_folder, filename)

            # Loga a figura no wandb
            s = "epoch {}".format(epoch-1)
            wandbfig = wandb.Image(fig, caption="Epoch:{}".format(epoch-1))
            wandb.log({s: wandbfig})

            if config.QUIET_PLOT:
                plt.close(fig)

        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            # Step de treinamento
            target = input_image
            gen_loss, disc_loss, gen_gan_loss, gen_l1_loss, disc_real_loss, disc_fake_loss = train_step_nodisc(generator, input_image, target)

           # Cálculo da acurácia
            y_real, y_pred, acc = utils.evaluate_accuracy(generator, disc, test_ds, y_real, y_pred)

            # Acrescenta a loss no arquivo
            loss_df = loss_df.append({"Loss G": gen_loss.numpy(), "Loss D" : disc_loss.numpy()}, ignore_index = True)
            # Log as métricas no wandb 
            wandb.log({ 'gen_loss': gen_loss.numpy(), 'gen_gan_loss': gen_gan_loss.numpy(), 'gen_l1_loss': gen_l1_loss.numpy(),
                        'disc_loss': disc_loss.numpy(), 'disc_real_loss': disc_real_loss.numpy(), 'disc_fake_loss': disc_fake_loss.numpy(),
                        'test_accuracy': acc})

            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every 20 epochs
        if (epoch) % config.CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
            
        # salva o arquivo de losses a cada época e plota como está ficando
        loss_df.to_csv(experiment_folder + "losses.csv")
        if not config.QUIET_PLOT:
            utils.plot_losses(loss_df)
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})
        
    dt = time.time() - t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  
   


#%% TESTA O CÓDIGO E MOSTRA UMA IMAGEM DO DATASET

inp = load(train_folder+'/male/000016.jpg')
# casting to int for matplotlib to show the image
if not config.QUIET_PLOT:
    plt.figure()
    plt.imshow(inp/255.0)

#%% PREPARAÇÃO DOS MODELOS

# Define se irá ter a restrição de tamanho de peso da WGAN (clipping)
constrained = False
if config.loss_type == 'wgan':
        constrained = True

# CRIANDO O MODELO DE GERADOR
if config.gen_model == 'resnet':
    generator = net.resnet_generator(config.IMG_SIZE)
elif config.gen_model == 'resnet_vetor': 
    generator = net.resnet_adapted_generator(config.IMG_SIZE)
elif config.gen_model == 'full_resnet':
    generator = net.VT_full_resnet_generator(config.IMG_SIZE)
elif config.gen_model == 'simple_decoder':
    generator = net.VT_simple_decoder(config.IMG_SIZE)
elif config.gen_model == 'full_resnet_dis':
    generator = net.VT_full_resnet_generator_disentangled(config.IMG_SIZE)
elif config.gen_model == 'simple_decoder_dis':
    generator = net.VT_simple_decoder_disentangled(config.IMG_SIZE)
elif config.gen_model == 'full_resnet_smooth':
    generator = net.VT_full_resnet_generator_smooth_disentangle(config.IMG_SIZE)
elif config.gen_model == 'simple_decoder_smooth':
    generator = net.VT_simple_decoder_smooth_disentangle(config.IMG_SIZE)
elif config.gen_model == 'transfer':
    generator = transfer.transfer_model(config.IMG_SIZE, config.transfer_generator_path, config.transfer_generator_filename, 
    config.transfer_middle_model, config.transfer_encoder_last_layer, config.transfer_decoder_first_layer, config.transfer_trainable,
    config.transfer_disentangle, config.transfer_smooth_vector)
elif config.gen_model == 'encoder_decoder':
    encoder = net.resnet_encoder(config.IMG_SIZE)
    decoder = net.resnet_decoder(config.IMG_SIZE)
else:
    raise utils.GeneratorError(config.gen_model)

# CRIANDO O MODELO DE DISCRIMINADOR
if config.ADVERSARIAL:
    if config.disc_model == 'patchgan':
        disc = net.patchgan_discriminator(config.IMG_SIZE)
    elif config.disc_model == 'stylegan_adapted': 
        disc = net.stylegan_discriminator_patchgan(config.IMG_SIZE)
    elif config.disc_model == 'stylegan':
        disc = net.stylegan_discriminator(config.IMG_SIZE, constrained = constrained)
    else:
        raise utils.DiscriminatorError(config.disc_model)

# Define os otimizadores
if config.USE_FULL_GENERATOR: 
    generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1=config.ADAM_BETA_1)
else:
    encoder_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1=config.ADAM_BETA_1)
    decoder_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1=config.ADAM_BETA_1)

if config.ADVERSARIAL:  
    discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1=config.ADAM_BETA_1)

#%% EXECUÇÃO

# Prepara os inputs
train_dataset = tf.data.Dataset.list_files(train_folder+'*/*.jpg')
train_size = len(list(train_dataset))
train_dataset = train_dataset.map(load_image_train)
train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
train_dataset = train_dataset.batch(config.BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(test_folder+'*/*.jpg')
test_size = len(list(test_dataset))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(config.BATCH_SIZE)

# Prepara o checkpoint
if config.ADVERSARIAL:
    if config.USE_FULL_GENERATOR: 
        # Prepara o checkpoint (gerador)
        checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                        discriminator_optimizer = discriminator_optimizer,
                                        generator = generator,
                                        disc = disc)
    else:
        # Prepara o checkpoint (encoder/decoder)
        checkpoint = tf.train.Checkpoint(encoder_optimizer = encoder_optimizer,
                                    decoder_optimizer = decoder_optimizer,
                                    discriminator_optimizer = discriminator_optimizer,
                                    encoder = encoder,
                                    decoder = decoder,
                                    disc = disc)
else:
    # Prepara o checkpoint (nodisc)
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
if config.USE_FULL_GENERATOR: 
    generator.save(model_folder+'ae_generator.h5')
else:
    encoder.save(model_folder+'ae_encoder.h5')
    decoder.save(model_folder+'ae_decoder.h5')

if config.ADVERSARIAL:
    disc.save(model_folder+'ae_discriminator.h5')


#%% TREINAMENTO

if config.FIRST_EPOCH <= config.EPOCHS:
    
    if config.ADVERSARIAL:
        if config.USE_FULL_GENERATOR: 
            fit(generator, disc, train_dataset, config.FIRST_EPOCH, config.EPOCHS, test_dataset)
        elif (not config.USE_FULL_GENERATOR):
            fit_encdec(encoder, decoder, disc, train_dataset, config.FIRST_EPOCH, config.EPOCHS, test_dataset)
        else:
            raise utils.LossError(config.loss_type)
    else:
        fit_nodisc(generator, train_dataset, config.FIRST_EPOCH, config.EPOCHS, test_dataset)

#%% VALIDAÇÃO

## Após o treinamento, loga uma imagem do dataset de teste para ver como ficou
for example_input in test_dataset.take(1):
    filename = "epoch_" + str(config.EPOCHS).zfill(len(str(config.EPOCHS))) + ".jpg"
    fig = utils.generate_save_images_gen(generator, example_input, result_folder, filename)

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
        if config.USE_FULL_GENERATOR: 
            utils.generate_save_images_gen(generator, img, result_test_folder, filename)
        else:
            utils.generate_save_images(encoder, decoder, img, result_test_folder, filename)
        c = c + 1
if config.QUIET_PLOT:
    plt.close("all")
        
## Plota as losses
try: 
    loss_df = pd.read_csv(experiment_folder + "losses.csv")
    fig = utils.plot_losses(loss_df)

    if config.QUIET_PLOT:
        plt.close(fig)
except:
    None

## Salva os modelos 
if config.USE_FULL_GENERATOR: 
    generator.save(model_folder+'ae_generator.h5')
else:
    encoder.save(model_folder+'ae_encoder.h5')
    decoder.save(model_folder+'ae_decoder.h5')

disc.save(model_folder+'ae_discriminator.h5')

# Salva os hiperparametros utilizados num arquivo txt
f = open(experiment_folder + "parameters.txt","w+")
f.write("LAMBDA = " + str(config.LAMBDA) + "\n")
f.write("BATCH_SIZE = " + str(config.BATCH_SIZE) + "\n")
f.write("BUFFER_SIZE = " + str(config.BUFFER_SIZE) + "\n")
f.write("IMG_SIZE = " + str(config.IMG_SIZE) + "\n")
f.write("EPOCHS = " + str(config.EPOCHS) + "\n")

f.write("LEARNING_RATE = " + str(config.LEARNING_RATE) + "\n")
f.write("ADAM_BETA_1 = " + str(config.ADAM_BETA_1) + "\n")

f.write("CHECKPOINT_EPOCHS = " + str(config.CHECKPOINT_EPOCHS) + "\n")
f.write("LOAD_CHECKPOINT = " + str(config.LOAD_CHECKPOINT) + "\n")
f.write("FIRST_EPOCH = " + str(config.FIRST_EPOCH) + "\n")
f.write("NUM_TEST_PRINTS = " + str(config.NUM_TEST_PRINTS) + "\n")
f.write("LAMBDA_GP = " + str(config.LAMBDA_GP) + "\n")
f.write("\n")
f.write("gen_model = " + str(config.gen_model) + "\n")
f.write("disc_model = " + str(config.disc_model) + "\n")
f.write("loss_type = " + str(config.loss_type) + "\n")
f.write("USE_FULL_GENERATOR = " + str(config.USE_FULL_GENERATOR) + "\n")

f.close()
wandb.finish()