### Imports
import os
import time
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# Módulos próprios
import networks as net
import utils

### Tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)

import tensorflow as tf

# Evita o erro "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Verifica se a GPU está disponível:
print(tf.config.list_physical_devices('GPU'))
# Verifica se a GPU está sendo usada na sessão
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(sess)

#%% HIPERPARÂMETROS
LAMBDA = 100
BATCH_SIZE = 1
BUFFER_SIZE = 150
IMG_SIZE = 256

EPOCHS = 3
CHECKPOINT_EPOCHS = 1
LOAD_CHECKPOINT = True
FIRST_EPOCH = 1
NUM_TEST_PRINTS = 10
KEEP_CHECKPOINTS = 2
LAMBDA_GP = 10 # Intensidade do Gradient Penalty da WGAN-GP
# WGAN_NCRITIC = 1 # A StyleGAN treina o Discriminador e o Gerador na mesma proporção (Ncritic = 1)

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
exp = "11A"

# Modelo do gerador. Possíveis = 'resnet', 'resnet_vetor', 'encoder_decoder', 'full_resnet', 'simple_decoder', 'full_resnet_dis', 'simple_decoder_dis'
gen_model = 'resnet'

# Modelo do discriminador. Possíveis = 'patchgan', 'patchgan_adapted', 'stylegan_adapted', 'stylegan'
disc_model = 'stylegan'

# Tipo de loss. Possíveis = 'patchganloss', 'wgan', 'wgan-gp'
loss_type = 'wgan'

# Acerta o flag USE_FULL_GENERATOR que indica se o gerador é único (full) ou partido (encoder + decoder)
if gen_model == 'encoder_decoder':
    USE_FULL_GENERATOR = False
else:
    USE_FULL_GENERATOR = True
    
# Valida se pode ser usado o tipo de loss com o tipo de discriminador
if loss_type == 'patchganloss':
    if not(disc_model == 'patchgan' or disc_model == 'patchgan_adapted' or  disc_model == 'stylegan_adapted'):
        raise utils.LossCompatibilityError(loss_type, disc_model)
if loss_type == 'wgan' or loss_type == 'wgan-gp':
    if not(disc_model == 'stylegan'):
        raise utils.LossCompatibilityError(loss_type, disc_model)


#%% Prepara as pastas

base_root = "../"

### Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Experimentos/'
experiment_folder = experiment_root + 'EXP' + exp + '_'
experiment_folder += 'gen_'
experiment_folder += gen_model
experiment_folder += '_disc_'
experiment_folder += disc_model
#experiment_folder += '_loss_'
#experiment_folder += loss_type
if BATCH_SIZE != 1:
    experiment_folder += '_BATCH_' + str(BATCH_SIZE)
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
    cropped_image = tf.image.random_crop(value = input_image, size = [IMG_SIZE, IMG_SIZE, 3])
    return cropped_image

# normalizing the images to [-1, 1]
def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

# Equivalente a random_jitter = tf.function(random.jitter)
@tf.function()
def random_jitter(input_image):
    # resizing to 286 x 286 x 3
    input_image = resize(input_image, 286, 286)
    
    # randomly cropping to 256 x 256 x 3
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
    input_image = resize(input_image, IMG_SIZE, IMG_SIZE)
    input_image = normalize(input_image)
    return input_image


#%% DEFINIÇÃO DAS LOSSES

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
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def loss_patchgan_discriminator(disc_real_output, disc_generated_output):
    # Ld = RealLoss + FakeLoss
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = BCE(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = BCE(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


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
    loss_g = gan_loss + (LAMBDA * l1_loss)
    return loss_g

def loss_wgan_discriminator(disc_real_output, disc_generated_output):
    # Maximizar E(D(x_real)) - E(D(x_fake)) é equivalente a minimizar -(E(D(x_real)) - E(D(x_fake))) ou E(D(x_fake)) -E(D(x_real))
    loss_d = tf.reduce_mean(disc_generated_output) - tf.reduce_mean(disc_real_output)
    return loss_d


'''
Wasserstein GAN - Gradient Penalty (WGAN-GP): A WGAN tem uma forma muito bruta de assegurar a continuidade de Lipschitz, então
os autores criaram o conceito de Gradient Penalty para manter essa condição de uma forma mais suave.
- O gerador tem a MESMA loss da WGAN
- O discriminador, em vez de ter seus pesos limitados pelo clipping, ganham uma penalidade de gradiente que deve ser calculada
'''
def loss_wgangp_generator(disc_generated_output, gen_output, target):
    return loss_wgan_generator(disc_generated_output, gen_output, target)

def loss_wgangp_discriminator(disc, disc_real_output, disc_generated_output, real_img, generated_img, target):
    loss_wgan = loss_wgan_discriminator(disc_real_output, disc_generated_output)
    gp = gradient_penalty_conditional(disc, real_img, generated_img, target)
    loss_d = loss_wgan + LAMBDA_GP * gp
    return loss_d

def gradient_penalty(disc, real_img, generated_img):
    """ 
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    From: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb#scrollTo=LhzOUkhYSOPG
    """
    # Get the interpolated image
    alpha = tf.random.normal([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
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
    """ 
    Adapted to Conditional Discriminators
    Calculates the gradient penalty.
    This loss is calculated on an interpolated image and added to the discriminator loss.
    From: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb#scrollTo=LhzOUkhYSOPG
    """
    # Get the interpolated image
    alpha = tf.random.normal([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
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
PATCHGAN - GERADOR ÚNICO
'''

# Função para o gerador
@tf.function
def train_step_patchgan(generator, disc, input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        gen_image = generator(input_image, training = True)
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)
          
        gen_total_loss, gen_gan_loss, gen_l1_loss = loss_patchgan_generator(disc_gen, gen_image, target)
        disc_loss = loss_patchgan_discriminator(disc_real, disc_gen)
    
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_total_loss, disc_loss)

    
# Função para o gerador
def fit_patchgan(generator, disc, train_ds, first_epoch, epochs, test_ds):
    
    # Lê arquivo com as losses
    try: 
        loss_df = pd.read_csv(experiment_folder + "losses.csv")
    except:
        loss_df = pd.DataFrame(columns = ["Loss G", "Loss D"])
    
    t0 = time.time()
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            utils.generate_save_images_gen(generator, example_input, result_folder, filename)
        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            target = input_image
            gen_loss, disc_loss = train_step_patchgan(generator, disc, input_image, target, epoch)
            
            # Acrescenta a loss no arquivo
            loss_df = loss_df.append({"Loss G": gen_loss.numpy(), "Loss D" : disc_loss.numpy()}, ignore_index = True)
            
            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                # utils.plot_losses(loss_df)
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every 20 epochs
        if (epoch) % CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
            
        # salva o arquivo de losses a cada época e plota como está ficando
        loss_df.to_csv(experiment_folder + "losses.csv")
        utils.plot_losses(loss_df)
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        
    dt = time.time() - t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  
    

'''
PATCHGAN - GERADOR SEPARADO EM ENCODER / DECODER
'''
    
# Função para o encoder/decoder
@tf.function
def train_step_patchgan_encdec(encoder, decoder, disc, input_image, target, epoch):
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
        
        latent = encoder(input_image, training = True)
        gen_image = decoder(latent, training = True) 
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)
          
        gen_total_loss, gen_gan_loss, gen_l1_loss = loss_patchgan_generator(disc_gen, gen_image, target)
        disc_loss = loss_patchgan_discriminator(disc_real, disc_gen)
    
    encoder_gradients = enc_tape.gradient(gen_total_loss, encoder.trainable_variables)
    decoder_gradients = dec_tape.gradient(gen_total_loss, decoder.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
    decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_total_loss, disc_loss)

# Função para o encoder/decoder
def fit_patchgan_encdec(encoder, decoder, disc, train_ds, first_epoch, epochs, test_ds):
    
    # Lê arquivo com as losses
    try: 
        loss_df = pd.read_csv(experiment_folder + "losses.csv")
    except:
        loss_df = pd.DataFrame(columns = ["Loss G", "Loss D"])
    
    t0 = time.time()
    
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            utils.generate_save_images(encoder, decoder, example_input, result_folder, filename)
        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            target = input_image
            gen_loss, disc_loss = train_step_patchgan_encdec(encoder, decoder, disc, input_image, target, epoch)
            
            # Acrescenta a loss no arquivo
            loss_df = loss_df.append({"Loss G": gen_loss.numpy(), "Loss D" : disc_loss.numpy()}, ignore_index = True)
            
            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every x epochs
        if (epoch) % CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
            
        # salva o arquivo de losses a cada época e plota como está ficando
        loss_df.to_csv(experiment_folder + "losses.csv")
        utils.plot_losses(loss_df)
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
    
    dt = time.time()-t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  


'''
WGAN - GERADOR ÚNICO
'''

# Função para o gerador
@tf.function
def train_step_wgan(generator, disc, input_image, target, epoch):
    # Está implementada sem o WGAN_NCRITIC
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        gen_image = generator(input_image, training = True)
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)
          
        gen_loss = loss_wgan_generator(disc_gen, gen_image, target)
        disc_loss = loss_wgan_discriminator(disc_real, disc_gen)
    
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_loss, disc_loss)

# Função para o gerador
def fit_wgan(generator, disc, train_ds, first_epoch, epochs, test_ds):
    
    # Lê arquivo com as losses
    try: 
        loss_df = pd.read_csv(experiment_folder + "losses.csv")
    except:
        loss_df = pd.DataFrame(columns = ["Loss G", "Loss D"])
    
    t0 = time.time()
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            utils.generate_save_images_gen(generator, example_input, result_folder, filename)
        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            target = input_image
            gen_loss, disc_loss = train_step_wgan(generator, disc, input_image, target, epoch)
            
            # Acrescenta a loss no arquivo
            loss_df = loss_df.append({"Loss G": gen_loss.numpy(), "Loss D" : disc_loss.numpy()}, ignore_index = True)
            
            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                # utils.plot_losses(loss_df)
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every x epochs
        if (epoch) % CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
            
        # salva o arquivo de losses a cada época e plota como está ficando
        loss_df.to_csv(experiment_folder + "losses.csv")
        utils.plot_losses(loss_df)
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        
    dt = time.time() - t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  
    

'''
WGAN-GP - GERADOR ÚNICO
'''

# Função para o gerador
@tf.function
def train_step_wgangp(generator, disc, input_image, target, epoch):
    # Está implementada sem o WGAN_NCRITIC
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        gen_image = generator(input_image, training = True)
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)
          
        gen_loss = loss_wgangp_generator(disc_gen, gen_image, target)
        disc_loss = loss_wgangp_discriminator(disc, disc_real, disc_gen, input_image, gen_image, target)
    
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_loss, disc_loss)

# Função para o gerador
def fit_wgangp(generator, disc, train_ds, first_epoch, epochs, test_ds):
    
    # Lê arquivo com as losses
    try: 
        loss_df = pd.read_csv(experiment_folder + "losses.csv")
    except:
        loss_df = pd.DataFrame(columns = ["Loss G", "Loss D"])
    
    t0 = time.time()
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        
        for example_input in test_ds.take(1):
            filename = "epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            utils.generate_save_images_gen(generator, example_input, result_folder, filename)
        print("Epoch: ", epoch)
        
        # Train
        for n, input_image in train_ds.enumerate():
            
            target = input_image
            gen_loss, disc_loss = train_step_wgangp(generator, disc, input_image, target, epoch)
            
            # Acrescenta a loss no arquivo
            loss_df = loss_df.append({"Loss G": gen_loss.numpy(), "Loss D" : disc_loss.numpy()}, ignore_index = True)
            
            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every 20 epochs
        if (epoch) % CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
            
        # salva o arquivo de losses a cada época e plota como está ficando
        loss_df.to_csv(experiment_folder + "losses.csv")
        utils.plot_losses(loss_df)
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        
    dt = time.time() - t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  


#%% TESTA O CÓDIGO E MOSTRA UMA IMAGEM DO DATASET

inp = load(train_folder+'/male/000016.jpg')
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp/255.0)

#%% PREPARAÇÃO DOS MODELOS

# Define se irá ter a restrição de tamanho de peso da WGAN (clipping)
constrained = False
if loss_type == 'wgan':
        constrained = True

# CRIANDO O MODELO DE GERADOR
if gen_model == 'resnet':
    generator = net.resnet_generator()
elif gen_model == 'resnet_vetor': 
    generator = net.resnet_adapted_generator()
elif gen_model == 'full_resnet':
    generator = net.VT_full_resnet_generator()
elif gen_model == 'simple_decoder':
    generator = net.VT_simple_decoder()
elif gen_model == 'full_resnet_dis':
    generator = net.VT_full_resnet_generator_disentangled()
elif gen_model == 'simple_decoder_dis':
    generator = net.VT_simple_decoder_disentangled()
elif gen_model == 'encoder_decoder':
    encoder = net.resnet_encoder()
    decoder = net.resnet_decoder()
else:
    raise utils.GeneratorError(gen_model)
    
# CRIANDO O MODELO DE DISCRIMINADOR
if disc_model == 'patchgan':
    disc = net.patchgan_discriminator()
elif disc_model == 'patchgan_adapted': 
    disc = net.patchgan_discriminator_adapted()
elif disc_model == 'stylegan_adapted': 
    disc = net.stylegan_discriminator_patchgan()
elif disc_model == 'stylegan':
    disc = net.stylegan_discriminator(constrained = constrained)
else:
    raise utils.DiscriminatorError(disc_model)

# Define os otimizadores
if USE_FULL_GENERATOR: 
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
else:
    encoder_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    decoder_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#%% EXECUÇÃO

# Prepara os inputs
train_dataset = tf.data.Dataset.list_files(train_folder+'*/*.jpg')
train_size = len(list(train_dataset))
train_dataset = train_dataset.map(load_image_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(test_folder+'*/*.jpg')
test_size = len(list(test_dataset))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


if USE_FULL_GENERATOR: 
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

# Se for o caso, recupera o checkpoint mais recente
if LOAD_CHECKPOINT:
    print("Carregando checkpoint mais recente...")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint != None:
        checkpoint.restore(latest_checkpoint)
        FIRST_EPOCH = int(latest_checkpoint.split("-")[1]) + 1
    else:
        FIRST_EPOCH = 1
        
        
# Salva o gerador e o discriminador (principalmente para visualização)
if USE_FULL_GENERATOR: 
    generator.save(model_folder+'ae_generator.h5')
else:
    encoder.save(model_folder+'ae_encoder.h5')
    decoder.save(model_folder+'ae_decoder.h5')
disc.save(model_folder+'ae_discriminator.h5')

#%% TREINAMENTO

if FIRST_EPOCH <= EPOCHS:
    
    # CONDIÇÕES DA LOSS PATCHGAN
    if USE_FULL_GENERATOR and loss_type == 'patchganloss': 
        fit_patchgan(generator, disc, train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
    elif (not USE_FULL_GENERATOR) and loss_type == 'patchganloss':
        fit_patchgan_encdec(encoder, decoder, disc, train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
    
    # CONDIÇÕES DA LOSS WGAN
    elif USE_FULL_GENERATOR and loss_type == 'wgan':
        fit_wgan(generator, disc, train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
    #elif (not USE_FULL_GENERATOR) and loss_type == 'wgan':
    #    fit_wgan_encdec(encoder, decoder, disc, train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
        
    # CONDIÇÕES DA LOSS WGAN-GP
    elif USE_FULL_GENERATOR and loss_type == 'wgan-gp':
        fit_wgangp(generator, disc, train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
    #elif (not USE_FULL_GENERATOR) and loss_type == 'wgan-gp':
    #    fit_wgangp_encdec(encoder, decoder, disc, train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
    
    else:
        raise utils.LossError(loss_type)

#%% VALIDAÇÃO

## Gera imagens do dataset de teste
c = 1
if NUM_TEST_PRINTS > 0:
    for img in test_dataset.take(NUM_TEST_PRINTS):
        filename = "test_results_" + str(c).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
        if USE_FULL_GENERATOR: 
            utils.generate_save_images_gen(generator, img, result_test_folder, filename)
        else:
            utils.generate_save_images(encoder, decoder, img, result_test_folder, filename)
        
        c = c + 1
        
## Plota as losses
try: 
    loss_df = pd.read_csv(experiment_folder + "losses.csv")
    utils.plot_losses(loss_df)
except:
    None

## Salva os modelos 
if USE_FULL_GENERATOR: 
    generator.save(model_folder+'ae_generator.h5')
else:
    encoder.save(model_folder+'ae_encoder.h5')
    decoder.save(model_folder+'ae_decoder.h5')

disc.save(model_folder+'ae_discriminator.h5')


# Salva os hiperparametros utilizados num arquivo txt
f = open(experiment_folder + "parameters.txt","w+")
f.write("LAMBDA = " + str(LAMBDA) + "\n")
f.write("BATCH_SIZE = " + str(BATCH_SIZE) + "\n")
f.write("BUFFER_SIZE = " + str(BUFFER_SIZE) + "\n")
f.write("IMG_SIZE = " + str(IMG_SIZE) + "\n")
f.write("EPOCHS = " + str(EPOCHS) + "\n")
f.write("CHECKPOINT_EPOCHS = " + str(CHECKPOINT_EPOCHS) + "\n")
f.write("LOAD_CHECKPOINT = " + str(LOAD_CHECKPOINT) + "\n")
f.write("FIRST_EPOCH = " + str(FIRST_EPOCH) + "\n")
f.write("NUM_TEST_PRINTS = " + str(NUM_TEST_PRINTS) + "\n")
f.write("LAMBDA_GP = " + str(LAMBDA_GP) + "\n")
# f.write("WGAN_NCRITIC = " + str(WGAN_NCRITIC) + "\n")
f.write("\n")
f.write("gen_model = " + str(gen_model) + "\n")
f.write("disc_model = " + str(disc_model) + "\n")
f.write("loss_type = " + str(loss_type) + "\n")
f.write("USE_FULL_GENERATOR = " + str(USE_FULL_GENERATOR) + "\n")
# f.write("\n")
# f.write("Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n".format(EPOCHS, dt/60, dt))

f.close()