import tensorflow as tf
import tensorflow_addons as tfa

## Tipo de normalização
# norm_layer = tf.keras.layers.BatchNormalization
norm_layer = tfa.layers.InstanceNormalization

## Modo de inicialização dos pesos
initializer = tf.random_normal_initializer(0., 0.02)



#%%

def VT_full_resnet_generator():

    '''
    Adaptado com base no gerador Resnet da Pix2Pix
    Feito de forma a gerar um vetor latente entre o encoder e o decoder, mas o decoder é também um resnet
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
    
    # Criação do vetor latente 
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Upsamplings
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block_transpose(x, 256)
    
    # Reconstrução pós blocos residuais
    
    #--
    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    #--
    x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[1, 1],[1, 1]])(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def VT_full_resnet_generator_disentangled():

    '''
    Adaptado com base no gerador Resnet da Pix2Pix
    Feito de forma a gerar um vetor latente entre o encoder e o decoder, mas o decoder é também um resnet
    Após o vetor latente, usar 8 camadas Dense para "desembaraçar" o espaço latente, como feito na StyleGAN
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
    
    # Criação do vetor latente 
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Disentanglement (de z para w, StyleGAN)
    for i in range(8):
        x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Upsamplings
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block_transpose(x, 256)
    
    # Reconstrução pós blocos residuais
    
    #--
    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    #--
    x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[1, 1],[1, 1]])(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def VT_simple_decoder():

    '''
    Adaptado com base no gerador Resnet da Pix2Pix
    Feito de forma a gerar um vetor latente entre o encoder e o decoder, mas o decoder é também um resnet
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
    
    # Criação do vetor latente 
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Upsamples
    # Todos os upsamples vão ser feitos com o simple_upsample, seguidos de duas convoluções na mesma dimensão
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 2, 2, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 4, 4, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 8, 8, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 16, 16, 512
    x = VT_simple_upsample_block(x, 256, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 32, 32, 256
    x = VT_simple_upsample_block(x, 128, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 64, 64, 128
    x = VT_simple_upsample_block(x, 64, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 128, 128, 64
    x = VT_simple_upsample_block(x, 32, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 256, 256, 32

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[1, 1],[1, 1]])(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def VT_simple_decoder_disentangled():

    '''
    Adaptado com base no gerador Resnet da Pix2Pix
    Feito de forma a gerar um vetor latente entre o encoder e o decoder, mas o decoder é também um resnet
    Após o vetor latente, usar 8 camadas Dense para "desembaraçar" o espaço latente, como feito na StyleGAN
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
    
    # Criação do vetor latente 
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Disentanglement (de z para w, StyleGAN)
    for i in range(8):
        x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Upsamples
    # Todos os upsamples vão ser feitos com o simple_upsample, seguidos de duas convoluções na mesma dimensão
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 2, 2, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 4, 4, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 8, 8, 512
    x = VT_simple_upsample_block(x, 512, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 16, 16, 512
    x = VT_simple_upsample_block(x, 256, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 32, 32, 256
    x = VT_simple_upsample_block(x, 128, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 64, 64, 128
    x = VT_simple_upsample_block(x, 64, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 128, 128, 64
    x = VT_simple_upsample_block(x, 32, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear') #--- 256, 256, 32

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[1, 1],[1, 1]])(x)
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    # print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


#%% MODELOS ORIGINAIS

def resnet_encoder():
    
    '''
    Adaptado do gerador utilizado nos papers Pix2Pix e CycleGAN
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)

    # Camadas finais
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = 512, activation = 'softmax')(x)    

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def resnet_decoder():
    
    '''
    Adaptado do gerador utilizado nos papers Pix2Pix e CycleGAN
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [512])
    x = tf.expand_dims(inputs, axis = 1)
    x = tf.expand_dims(x, axis = 1)
    
    # Reconstrução da imagem
    x = upsample(x, 512)
    x = upsample(x, 512)
    x = upsample(x, 512)
    x = upsample(x, 512)
    x = upsample(x, 256)
    x = upsample(x, 128)
    x = upsample(x, 64)
    
    # Última camada
    x = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)    


def resnet_generator():
    
    '''
    Versão original do gerador utilizado nos papers Pix2Pix e CycleGAN
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
        
    # print(x.shape)
        
    # Reconstrução da imagem
    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    x = tf.keras.layers.ZeroPadding2D([[2, 2],[2, 2]])(x)
    x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (7, 7) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    #print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def resnet_adapted_generator():
    
    '''
    Adaptado do gerador utilizado nos papers Pix2Pix e CycleGAN
    Modificado de forma a gerar um vetor latente entre o encoder e o decoder
    '''
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [256 , 256 , 3])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(9):
        x = resnet_block(x, 256)
        
    # Criação do vetor latente
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Dense(units = 512, kernel_initializer=initializer)(x)
    
    # Criação do vetor latente (alternativa)
    vecsize = 512
    x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = vecsize, kernel_initializer=initializer)(x)
    
    # Transforma novamente num tensor de terceira ordem
    x = tf.expand_dims(x, axis = 1)
    x = tf.expand_dims(x, axis = 1)
        
    # Reconstrução da imagem
    for i in range(5):
        x = tf.keras.layers.Conv2DTranspose(filters = vecsize, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
        x = norm_layer()(x)
        x = tf.keras.layers.ReLU()(x)
        
    x = tf.keras.layers.Conv2DTranspose(filters = vecsize/2, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = vecsize/4, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = vecsize/8, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    # x = tf.keras.layers.ZeroPadding2D([[2, 2],[2, 2]])(x)
    x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (7, 7) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    # print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)


def patchgan_discriminator():
    
    '''
    Versão original do discriminador utilizado nos papers Pix2Pix e CycleGAN
    '''
    
    # Inicializa a rede e os inputs
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    # Na implementação em torch, a concatenação ocorre dentro da classe pix2pixmodel
    x = tf.keras.layers.concatenate([inp, tar]) 
    
    # Convoluções
    x = tf.keras.layers.Conv2D(64, 4, strides=2, kernel_initializer=initializer, padding = 'same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(128, 4, strides=2, kernel_initializer=initializer, padding = 'valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(256, 4, strides=2, kernel_initializer=initializer, padding = 'valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, padding = 'same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    # Camada final (30 x 30 x 1) - Para usar o L1 loss
    x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, padding = 'same')(x)
    
    return tf.keras.Model(inputs=[inp, tar], outputs=x)


def patchgan_discriminator_adapted():
    
    '''
    Adaptado do discriminador utilizado nos papers Pix2Pix e CycleGAN
    Veio do tutorial Pix2Pix do Tensorflow
    '''
    
    # Inicializa a rede e os inputs
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    
    # Camadas intermediárias
    x = downsample(x, 64) # (bs, 128, 128, 64)
    x = downsample(x, 128) # (bs, 64, 64, 128)
    x = downsample(x, 256) # (bs, 32, 32, 256)
    x = tf.keras.layers.ZeroPadding2D()(x) # (bs, 34, 34, 256)
    
    # Camada de ajuste
    x = tf.keras.layers.Conv2D(512, 3, strides=1, kernel_initializer=initializer)(x) # (bs, 31, 31, 512)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x) # (bs, 33, 33, 512)
    
    # Camada final (30 x 30 x 1) - Para usar o L1 loss
    x = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer)(x) # (bs, 30, 30, 1)
    
    return tf.keras.Model(inputs=[inp, tar], outputs=x)


def stylegan_discriminator_patchgan():

    '''
    Adaptado do discriminador utilizado nos papers ProgGAN e styleGAN
    1ª adaptação é para poder usar a mesma loss e estrutura da PatchGAN, para aprendizado supervisionado
    2ª adaptação é para usar imagens 256x256:
        As primeiras 3 convoluições são mantidas (filters = 16, 16, 32) com as dimensões 256 x 256
        Então "pula" para a sexta convolução, que já é originalmente de tamanho 256 x 256 e continua daí para a frente
    3ª adaptação é para terminar com imagens 30x30, para poder usar a mesma loss da PatchGAN
    '''
    
    # Inicializa a rede e os inputs
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])    

    # Primeiras três convoluções adaptadas para 256x256
    x = tf.keras.layers.Conv2D(16, (1 , 1), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 16, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(16, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 16, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(32, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 32, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    
    # Etapa 256 (convoluções 6 e 7)
    x = tf.keras.layers.Conv2D(64, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 64, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 128, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 128, 128, 128)
    
    # Etapa 128
    x = tf.keras.layers.Conv2D(128, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 128, 128, 128)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 256, 128, 128)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 256, 64, 64)
    
    # Etapa 64
    x = tf.keras.layers.Conv2D(256, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 256, 64, 64)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 512, 64, 64)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 32, 32)
    
    # Etapa 32 
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 512, 32, 32)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same')(x) # (bs, 512, 32, 32)
    x = tf.keras.layers.LeakyReLU()(x)
    
    # Adaptação para finalizar com 30x30    
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer)(x) # (bs, 30, 30, 1)
    
    return tf.keras.Model(inputs=[inp, tar], outputs=x)


def stylegan_discriminator(constrained = False):

    '''
    Adaptado do discriminador utilizado nos papers ProgGAN e styleGAN
    1ª adaptação é para poder fazer o treinamento supervisionado, mas com a loss adaptada da WGAN ou WGAN-GP
    2ª adaptação é para usar imagens 256x256:
        As primeiras 3 convoluições são mantidas (filters = 16, 16, 32) com as dimensões 256 x 256
        Então "pula" para a sexta convolução, que já é originalmente de tamanho 256 x 256 e continua daí para a frente
    '''
    ## Restrições para o discriminador (usado na WGAN original)
    constraint = tf.keras.constraints.MinMaxNorm(min_value = -0.01, max_value = 0.01)
    if constrained == False:
        constraint = None

    
    # Inicializa a rede e os inputs
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])    

    # Primeiras três convoluções adaptadas para 256x256
    x = tf.keras.layers.Conv2D(16, (1 , 1), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint = constraint)(x) # (bs, 16, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(16, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint = constraint)(x) # (bs, 16, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(32, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint = constraint)(x) # (bs, 32, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    
    # Etapa 256 (convoluções 6 e 7)
    x = tf.keras.layers.Conv2D(64, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 64, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 128, 256, 256)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 128, 128, 128)
    
    # Etapa 128
    x = tf.keras.layers.Conv2D(128, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 128, 128, 128)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 256, 128, 128)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 256, 64, 64)
    
    # Etapa 64
    x = tf.keras.layers.Conv2D(256, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 256, 64, 64)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 64, 64)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 32, 32)
    
    # Etapa 32 
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 32, 32)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 32, 32)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 16, 16)
    
    # Etapa 16
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 16, 16)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 16, 16)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 8, 8)
    
    # Etapa 8
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 8, 8)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 8, 8)
    x = tf.keras.layers.LeakyReLU()(x)
    x = simple_downsample(x, scale = 2) # (bs, 512, 4, 4)
    
    # Final - 4 para 1
    # Nesse ponto ele faz uma minibatch stddev. Avaliar depois fazer BatchNorm
    x = tf.keras.layers.Conv2D(512, (3 , 3), strides=1, kernel_initializer=initializer, padding = 'same', kernel_constraint=constraint)(x) # (bs, 512, 4, 4)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, (4 , 4), strides=1, kernel_initializer=initializer, kernel_constraint=constraint)(x) # (bs, 512, 1, 1)
    x = tf.keras.layers.LeakyReLU()(x)
    
    # Finaliza com uma Fully Connected 
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation = 'linear', kernel_constraint=constraint)(x)
    
    return tf.keras.Model(inputs=[inp, tar], outputs=x)


#%% BLOCOS 

def resnet_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet baseado na Resnet34
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x

def resnet_block_transpose(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet baseado na Resnet34, mas invertido (convoluções transpostas)
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


def resnet_bottleneck_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet bottleneck, baseado na Resnet50
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1))(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Terceira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x

def resnet_downsample_bottleneck_block(input_tensor, filters):
    
    ''' 
    Cria um bloco resnet bottleneck, com redução de dimensão, baseado na Resnet50
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''
    
    x = input_tensor
    skip = input_tensor
    
    # Convolução da skip connection
    skip = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(skip)
    skip = norm_layer()(skip)
    
    # Primeira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1))(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Terceira convolução (kernel = 1, 1)
    x = tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1))(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x

def upsample(x, filters):
    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x

def downsample(x, filters):
    # Reconstrução da imagem, baseada na Pix2Pix / CycleGAN    
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x

def simple_upsample(x, scale = 2, interpolation = 'bilinear'):
    # Faz um umpsample simplificado, baseado no Progressive Growth of GANs
    x = tf.keras.layers.UpSampling2D(size = (scale, scale), interpolation = interpolation)(x)
    return x

def VT_simple_upsample_block(x, filters, scale = 2, kernel_size = (3, 3), interpolation = 'bilinear'):
    
    x = simple_upsample(x, scale = scale, interpolation = interpolation) 
    
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x

def simple_downsample(x, scale = 2):
    # Faz um umpsample simplificado, baseado no Progressive Growth of GANs
    x = tf.keras.layers.AveragePooling2D(pool_size = (scale, scale))(x)
    return x

