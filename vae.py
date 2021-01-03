import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

output = "/content/Thesis/disentanglement_lib/data/og"

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 64 * 64*3
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


training_data = []

def create_training_data():
    for img in os.listdir(output):  
        try:
            img_array = cv2.imread(os.path.join(output,img))  # convert to array
            img_array = img_array/ 255
            training_data.append(img_array)  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass
        
create_training_data()
training_data = np.array(training_data)

class SamplingLayer(tf.keras.layers.Layer):
    def __init__(self):#, z_mean, z_log_sigma):
        super(SamplingLayer, self).__init__()
        #self.z_mean = z_mean
        #self.z_log_sigma = z_log_sigma
        
    def call(self, inputs):
        #epsilon = K.random_normal(shape=(K.shape(self.z_mean)[0], latent_dim), mean=0., stddev=0.1))
        z_mean, z_log_sigma = inputs
        return z_mean + K.exp(z_log_sigma) * K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.1)



original_dim = 64 * 64 * 1
intermediate_dim = 64
latent_dim = 10

encoder_inputs = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.MaxPool2D(pool_size=(2, 2), strides=None)(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)
# z = layers.Lambda(sampling)([z_mean, z_log_sigma])
z = SamplingLayer()([z_mean, z_log_sigma])
# Create encoder
encoder = keras.Model(encoder_inputs, [z_mean, z_log_sigma, z], name='encoder')
encoder.summary()





# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((16, 16, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()

X_train, X_test = train_test_split(training_data, test_size=0.2)


vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())


vae.fit(X_train, X_train,epochs=50,batch_size=32)



def plot_label_clusters(encoder, decoder, data):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(X_test)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1])
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    


plot_label_clusters(encoder, decoder, X_test)


def plot_label_clusters(encoder, decoder, data):
    # display a 2D plot of the digit classes in the latent space
    z = encoder.predict(test_img)
    return z
    

test_img = X_test[350].reshape(1,64,64,3)

z = plot_label_clusters(encoder, decoder, X_test)


def decode_embedding(z, decoder):
    return decoder.predict(z)

# reconstruct original image using latent space
ret = decode_embedding(z, decoder)
fig2 = plt.gcf()
plt.imshow(ret[0])
fig2.savefig('/content/Thesis/reconstructoin.jpg', dpi=100.7)


# reconstruct original image using latent space
img_array = cv2.imread("/content/Thesis/disentanglement_lib/9-750-5-1.png")# convert to array

test_img = img_array.reshape([1, 64, 64, 3]).astype(np.float32) / 255. # add this to our training_data 

z = plot_label_clusters(encoder, decoder, X_test)
ret = decode_embedding(z, decoder)
fig1 = plt.gcf()
plt.imshow(ret[0])

fig1.savefig('/content/Thesis/reconstructoin1.jpg', dpi=100.7)






def generate_new_images_vae(nb=10):
    plt.clf();
    f, ax = plt.subplots(2, nb//2, figsize=(20,7));
    for i in range(nb):
        z1 = np.random.rand(7,7)
        z1 =np.array([[-2.6344512,   2.9057171,  -3.166529  ,  0.03369812, -3.6336272,  -1.0407045,   3.183059,-1.6270628 ,1.8529456 ,1.6964235]])
        ret = decode_embedding(z1, decoder)
        ax[i%2][i//2].imshow(ret[0])
        ax[i%2][i//2].set_title('generated img {}'.format(i))
    ax[0][0].imshow(X_test[300])
    ax[0][0].set_title('training img')
    
generate_new_images_vae()

