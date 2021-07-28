# FUNÇÕES DE APOIO PARA O AUTOENCODER

import tensorflow as tf
import matplotlib.pyplot as plt

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
    plt.show()
    
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
    plt.show()
    
    f.savefig(save_destination + filename)