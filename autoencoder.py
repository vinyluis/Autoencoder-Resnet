### Imports
import os
import time
from matplotlib import pyplot as plt

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
BUFFER_SIZE = 200
IMG_SIZE = 256

EPOCHS = 5
CHECKPOINT_EPOCHS = 1
LOAD_CHECKPOINT = True
FIRST_EPOCH = 1
NUM_TEST_PRINTS = 10
KEEP_CHECKPOINTS = 2

### Controle do Modelo
# Modelo do gerador. Possíveis = 'resnet', 'resnet_vetor', 'encoder_decoder', 'full_resnet', 'simple_decoder'
gen_model = 'full_resnet'

# Modelo do discriminador. Possíveis = 'patchgan', 'patchgan_adapted', 'stylegan_adapted'
disc_model = 'stylegan_adapted'

# Acerta o flag USE_FULL_GENERATOR que indica se o gerador é único (full) ou partido (encoder + decoder)
if gen_model == 'encoder_decoder':
    USE_FULL_GENERATOR = False
else:
    USE_FULL_GENERATOR = True


#%% Prepara as pastas

### Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = '../Experimentos/'
experiment_folder = experiment_root + 'EXP_'
experiment_folder += 'gen_'
experiment_folder += gen_model
experiment_folder += '_disc_'
experiment_folder += disc_model
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
    
checkpoint_dir = experiment_folder + 'training_checkpoints'
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

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


'''
Perda do gerador
É uma perda de entropia cruzada sigmóide das imagens geradas e uma série de algumas .
O documento também inclui a perda de L1 que é MAE (erro médio absoluto) entre a imagem gerada e a imagem alvo.
Isso permite que a imagem gerada se torne estruturalmente semelhante à imagem de destino.
A fórmula para calcular a perda total do gerador = gan_loss + LAMBDA * l1_loss, onde LAMBDA = 100. Este valor foi decidido pelos autores do artigo .
O procedimento de treinamento para o gerador é mostrado abaixo:
'''

def generator_loss(disc_generated_output, gen_output, target):
    
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss


'''
Perda de discriminador

A função de perda do discriminador leva 2 entradas; imagens reais, imagens geradas
real_loss é uma perda de entropia cruzada sigmóide das imagens reais e uma matriz de uns (uma vez que estas são as imagens reais)
Generated_loss é uma perda de entropia cruzada sigmóide das imagens geradas e uma matriz de zeros (uma vez que estas são as imagens falsas)
Então o total_loss é a soma de real_loss e o generated_loss
'''

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss


#%% FUNÇÕES DO TREINAMENTO

'''
Para cada exemplo de entrada, gere uma saída.
O discriminador recebe a input_image e a imagem gerada como a primeira entrada. A segunda entrada é input_image e target_image.
Em seguida, calculamos a perda do gerador e do discriminador.
Em seguida, calculamos os gradientes de perda em relação às variáveis ​​do gerador e do discriminador (entradas) e os aplicamos ao otimizador.
Em seguida, registre as perdas no TensorBoard.
'''

# Função para o encoder/decoder
@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
        
        latent = encoder(input_image, training = True)
        gen_image = decoder(latent, training = True) 
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)
          
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_gen, gen_image, target)
        disc_loss = discriminator_loss(disc_real, disc_gen)
    
    encoder_gradients = enc_tape.gradient(gen_total_loss, encoder.trainable_variables)
    decoder_gradients = dec_tape.gradient(gen_total_loss, decoder.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
    decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss)

# Função para o gerador
@tf.function
def train_step_gen(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        gen_image= generator(input_image, training = True)
        
        disc_real = disc([input_image, target], training=True)
        disc_gen = disc([input_image, gen_image], training=True)
          
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_gen, gen_image, target)
        disc_loss = discriminator_loss(disc_real, disc_gen)
    
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss)

    
             
'''
O loop de treinamento real:

Repete o número de épocas.
Em cada época, ele limpa a tela e executa generate_images para mostrar seu progresso.
Em cada época, ele itera sobre o conjunto de dados de treinamento, imprimindo um '.' para cada exemplo.
Ele salva um ponto de verificação a cada 20 épocas.
'''

# Função para o encoder/decoder
def fit(train_ds, first_epoch, epochs, test_ds):
    
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
            train_step(input_image, target, epoch)
            
            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every 20 epochs
        if (epoch) % CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
    
    checkpoint.save(file_prefix = checkpoint_prefix)
    
    dt = time.time()-t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  
    
    
# Função para o gerador
def fit_gen(train_ds, first_epoch, epochs, test_ds):
    
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
            train_step_gen(input_image, target, epoch)
            
            # Printa pontinhos a cada 100. A cada 100 pontinhos, pula a linha
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()      
            
        # saving (checkpoint) the model every 20 epochs
        if (epoch) % CHECKPOINT_EPOCHS == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("\nSalvando checkpoint...")
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
    
    checkpoint.save(file_prefix = checkpoint_prefix)
    
    dt = time.time()-t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))  

  
#%% TESTA O CÓDIGO E MOSTRA UMA IMAGEM DO DATASET

inp = load(train_folder+'/male/000016.jpg')
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp/255.0)


#%% PREPARAÇÃO DOS MODELOS

# CRIANDO O MODELO DE GERADOR
class GeneratorError(Exception):
    def __init__(self, gen_model):
        print("O gerador " + gen_model + " é desconhecido")

if gen_model == 'resnet':
    generator = net.resnet_generator()
elif gen_model == 'resnet_vetor': 
    generator = net.resnet_adapted_generator()
elif gen_model == 'full_resnet':
    generator = net.VT_full_resnet_generator()
elif gen_model == 'simple_decoder':
    generator = net.VT_simple_decoder()
elif gen_model == 'encoder_decoder':
    encoder = net.resnet_encoder()
    decoder = net.resnet_decoder()
else:
    raise GeneratorError(gen_model)
    
# CRIANDO O MODELO DE DISCRIMINADOR
class DiscriminatorError(Exception):
    def __init__(self, disc_model):
        print("O discriminador " + disc_model + " é desconhecido")

if disc_model == 'patchgan':
    disc = net.patchgan_discriminator()
elif disc_model == 'patchgan_adapted': 
    disc = net.patchgan_discriminator_adapted()
elif disc_model == 'stylegan_adapted': 
    disc = net.stylegan_discriminator_adapted()
else:
    raise DiscriminatorError(disc_model)

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


# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = KEEP_CHECKPOINTS)

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
    if USE_FULL_GENERATOR: 
        fit_gen(train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)
    else:
        fit(train_dataset, FIRST_EPOCH, EPOCHS, test_dataset)

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
f.write("\n")
f.write("gen_model = " + str(gen_model) + "\n")
f.write("disc_model = " + str(disc_model) + "\n")
f.write("USE_FULL_GENERATOR = " + str(USE_FULL_GENERATOR) + "\n")
# f.write("\n")
# f.write("Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n".format(EPOCHS, dt/60, dt))

f.close()