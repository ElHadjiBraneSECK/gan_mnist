

import tensorflow as tf
from tqdm import tqdm
tf.__version__

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as k
from tensorflow.keras import layers
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images / 128) -1


train_images = np.expand_dims(train_images, axis=3)
train_images.shape


learning_rate = 1e-5
batch_size =  64
epochs = 500



sess = tf.Session()




x = tf.placeholder(tf.float32, (None, 28, 28,1))
z = tf.placeholder(tf.float32, (None, 100))





# function to generate noise
def get_noise(shape):
    return np.random.uniform(-1, 1, shape)


# Discriminator 

xin = layers.Input((1,28,28,1), tensor=x)

layer = layers.Conv2D(64, (5, 5), strides=(1,1),padding='same')(xin)
layer = layers.LeakyReLU()(layer)
layer = layers.Dropout(0.3)(layer)


layer = layers.Conv2D(128, (5, 5), strides=(2,2),padding='same')(layer)
layer = layers.LeakyReLU()(layer)
layer = layers.Dropout(0.3)(layer)


layer = layers.Flatten()(layer)
layer = layers.Dense(1)(layer)
layer = layers.Activation('sigmoid')(layer)

print("out_dec", layer.shape)
D = k.models.Model(xin, layer)



# Generator 

zin = layers.Input((100,), tensor=z)

layer = layers.Dense(7*7*256)(zin)
layer = layers.BatchNormalization()(layer)
layer = layers.LeakyReLU()(layer)

layer = layers.Reshape((7,7,256))(layer)

layer = layers.Conv2DTranspose(128, (5, 5), strides=(1,1),padding='same',use_bias=False)(layer)
layer = layers.BatchNormalization()(layer)
layer = layers.LeakyReLU()(layer)

layer = layers.Conv2DTranspose(64, (5, 5), strides=(2,2),padding='same',use_bias=False)(layer)
layer = layers.BatchNormalization()(layer)
layer = layers.LeakyReLU()(layer)

layer = layers.Conv2DTranspose(1, (5, 5), strides=(2,2),padding='same',use_bias=False)(layer)
layer = layers.Activation('tanh')(layer)

print("out_gen", layer.shape)

G = k.models.Model(zin, layer)



D_loss = -tf.log(D(x)) - tf.log(1-D(G(z)))
G_loss = -tf.log(D(G(z)))



D_opt = tf.train.AdamOptimizer(learning_rate=1e-5)
G_opt = tf.train.AdamOptimizer(learning_rate=1e-5)



D_grad = D_opt.minimize(D_loss, var_list=D.trainable_weights)
G_grad = G_opt.minimize(G_loss, var_list=G.trainable_weights)


sess.run(tf.initialize_all_variables())


D_loss_list = []
G_loss_list = []
for i in range (epochs):
    print("epoch "+str(i))
    np.random.shuffle(train_images)
    for j in tqdm(range(0, len(train_images), batch_size*2)):
        if len(train_images[j:j+batch_size*2])< batch_size*2:
            continue
        _, b = sess.run([D_grad, D_loss], feed_dict={z: get_noise((batch_size,100,)), x: train_images[j:j+batch_size]})
        _, c = sess.run([G_grad, G_loss], feed_dict={z: get_noise((batch_size,100,)), x: train_images[j+batch_size:j+batch_size*2]})
        D_loss_list.append(np.mean(b))
        G_loss_list.append(np.mean(c))
    
    print("D: ", str(D_loss_list[-1]), "G: ", str(G_loss_list[-1]))
        
    for j in range(2):
        img = sess.run(G(z), feed_dict={z: get_noise((1,100))})
        plt.imsave(str(i)+"_"+str(j)+".png", np.squeeze(img))
    
    plt.plot(D_loss_list,label="D_loss")
    plt.plot(G_loss_list,label="G_loss")
    plt.legend()
    plt.savefig("losses.png")
    plt.close()
    
    

sess.close()

