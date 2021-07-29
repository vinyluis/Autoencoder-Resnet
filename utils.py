# FUNÇÕES DE APOIO PARA O AUTOENCODER
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def generate_images(encoder, decoder, img_input):
    latent = encoder(img_input, training=True)
    img_predict = decoder(latent, training=True)
    plt.figure(figsize=(15,15))
    
    display_list = [img_input[0], img_predict[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()  

def generate_images_gen(generator, img_input):
    img_predict = generator(img_input, training=True)
    f = plt.figure(figsize=(15,15))
    
    display_list = [img_input[0], img_predict[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    f.show()
    
def generate_save_images(encoder, decoder, img_input, save_destination, filename):
    latent = encoder(img_input, training=True)
    img_predict = decoder(latent, training=True)
    f = plt.figure(figsize=(15,15))
    
    print("Latent Vector:")
    print(latent)
    
    display_list = [img_input[0], img_predict[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    f.show()
    
    f.savefig(save_destination + filename)
    
def generate_save_images_gen(generator, img_input, save_destination, filename):
    img_predict = generator(img_input, training=True)
    f = plt.figure(figsize=(15,15))
    
    display_list = [img_input[0], img_predict[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    f.show()
    
    f.savefig(save_destination + filename)

    return f


def plot_losses(loss_df, plot_ma = True, window = 100):
    
    # Plota o principal
    f = plt.figure()
    sns.lineplot(x = range(loss_df.shape[0]), y = loss_df["Loss G"])
    sns.lineplot(x = range(loss_df.shape[0]), y = loss_df["Loss D"])
    
    # Plota as médias móveis
    if plot_ma:
        
        lossG_ma = loss_df["Loss G"].rolling(window = window, min_periods = 1).mean()
        lossD_ma = loss_df["Loss D"].rolling(window = window, min_periods = 1).mean()
        sns.lineplot(x = range(loss_df.shape[0]), y = lossG_ma)
        sns.lineplot(x = range(loss_df.shape[0]), y = lossD_ma)
        plt.legend(["Loss G", "Loss D", "Loss G - MA", "Loss D - MA"])
    else:
        plt.legend(["Loss G", "Loss D"])
    
    f.show()
    
    return f

#%% TRATAMENTO DE EXCEÇÕES
    
class GeneratorError(Exception):
    def __init__(self, gen_model):
        print("O gerador " + gen_model + " é desconhecido")
    
class DiscriminatorError(Exception):
    def __init__(self, disc_model):
        print("O discriminador " + disc_model + " é desconhecido")
        
class LossError(Exception):
    def __init__(self, loss_type):
        print("A loss " + loss_type + " é desconhecida")
        
class LossCompatibilityError(Exception):
    def __init__(self, loss_type, disc_model):
        print("A loss " + loss_type + " não é compatível com o discriminador " + disc_model)

class SizeCompatibilityError(Exception):
    def __init__(self, img_size):
        print("IMG_SIZE " + img_size + " não está disponível")

class TransferUpsampleError(Exception):
    def __init__(self, upsample):
        print("Tipo de upsampling " + upsample + " não definido")