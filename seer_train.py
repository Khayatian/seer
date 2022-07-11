from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape, Concatenate, RepeatVector, Permute
from tensorflow.keras.layers import Activation, Conv2D
from tensorflow.keras.layers import ELU
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from keras.optimizers import Adam
import scipy.io
import numpy as np
import os


# One-hot-encode values into Boolean format
def one_hot_encode(y, lbl):
    z = np.zeros((len(y), lbl))  # number of labels
    idx = np.arange(len(y))
    z[idx, y] = 1
    return z


# Generator setup
def get_generator(weather_layer, condition_layer, noise_layer):
    # Prep operation condition
    hid = Concatenate()([weather_layer, condition_layer, noise_layer])
    hid = Dense(24 * 6 * 16)(hid)
    hid = ELU(alpha=1)(hid)
    hid = Dropout(0.4)(hid)
    hid = Reshape((24, 6, 16))(hid)
    # Convolution
    hid = Conv2D(16, kernel_size=(48, 12), strides=(1, 1), padding='same')(hid)  # to [6,6]
    hid = ELU(alpha=1)(hid)
    hid = Conv2D(8, kernel_size=(24, 6), strides=(1, 1), padding='same')(hid)  # to [12,6] (24 6)
    hid = ELU(alpha=1)(hid)
    hid = Conv2D(4, kernel_size=(12, 3), strides=(1, 1), padding='same')(hid)  # to [24,6] (12 3)
    hid = ELU(alpha=1)(hid)
    # Convolution with kernel size 1
    hid = Conv2D(1, (1, 1), padding="same")(hid)
    # Output
    out = Activation("tanh")(hid)
    # Compile model
    model = Model(inputs=[weather_layer, condition_layer, noise_layer], outputs=out)
    model.summary()
    return model, out


def get_discriminator(perf_layer, condition_layer, weather_layer):
    # Convolution
    hid1 = Conv2D(4, (48, 12), strides=(1, 1), padding='same')(perf_layer)  # 16
    hid1 = ELU(alpha=1)(hid1)
    hid2 = Conv2D(8, (12, 6), strides=(1, 1), padding='same')(perf_layer)   # 12
    hid2 = ELU(alpha=1)(hid2)
    hid3 = Conv2D(12, (6, 3), strides=(1, 1), padding='same')(perf_layer)   # 8
    hid3 = ELU(alpha=1)(hid3)
    hid4 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(perf_layer)  # 4
    hid4 = ELU(alpha=1)(hid4)
    # Concatenate
    hid = Concatenate()([hid1, hid2, hid3, hid4])
    # Flatten
    hid = Flatten()(hid)
    # Concatenate
    hid = Concatenate()([hid, condition_layer, weather_layer])
    # Logistic
    hid = Dense(1024)(hid)
    hid = ELU(alpha=1)(hid)
    hid = Dropout(0.4)(hid)
    # Output
    out = Dense(1, activation='sigmoid')(hid)  # activation='sigmoid'
    # Compile model
    model = Model(inputs=[perf_layer, condition_layer, weather_layer], outputs=out)
    model.summary()
    return model, out


# Generating latent noise
def generate_noise(n_samples, noise_dim):
    lat_noise = np.random.normal(0, 1, size=(n_samples, noise_dim))  # 0,1
    return lat_noise


# Associating noise to demand profiles
def generate_images(batch_id, noise_fuse):
    prfm = x_performance[batch_id * BATCH_SIZE: (batch_id + 1) * BATCH_SIZE] * \
           np.random.uniform(0.95 - noise_fuse, 1.05 + noise_fuse, size=(1, 1))  # 0.999 1.001
    return prfm


# Generating random operation labels
def generate_real_labels(batch_id, anmly):
    flse = generate_random_labels(int(anmly))
    lbls = y_operation[batch_id * BATCH_SIZE: (batch_id + 1) * BATCH_SIZE]
    flipped_lbls = np.random.choice(np.arange(len(lbls)), size=int(anmly))
    lbls[flipped_lbls, :] = flse
    return lbls


# Generating relevant operation labels
def generate_random_labels(n):
    y1 = np.random.choice(3, n)  # number of day type labels (Weekday, Saturday, Sunday/holiday)
    y1 = one_hot_encode(y1, 3)
    y2 = np.random.choice(2, n)  # number of daylight saving labels (daylight saving on/off)
    y2 = one_hot_encode(y2, 2)
    y = np.concatenate((y1, y2), axis=1)
    return y


# Load inputs
mat = scipy.io.loadmat('build9.mat')  # All inputs are stored in a Matlab data file named "build1.mat"
x_performance = mat['performance']  # In the file "build1.mat" , demand profiles are named "performance"
y_operation = mat['operation']  # In the file "build1.mat" , operation labels are named "performance"
z_weather = mat['weather']  # In the file "build1.mat" , weather profiles are named "performance"
n_train_samples = x_performance.shape[0]
x_performance = x_performance.reshape(n_train_samples, x_performance.shape[1], x_performance.shape[2], 1)  # reshaping


# Training properties
N_EPOCHS = 5000  # Between 5000 - 10000
BATCH_SIZE = n_train_samples  # Any value between 60 or 100 is a good batch size but depends on the GPU memory
num_batches = int(x_performance.shape[0] / BATCH_SIZE)  # Calculating the number of batches
epoch = 0

# Setting up the discriminator
demand_input = Input(shape=(24, 6, 1))  # input_1: shape of demand profiles (24 hours * 6 variables)
disc_operation_input = Input(shape=(5,))  # input_2: number of label categories (5)
disc_weather_input = Input(shape=(96,))  # input_x: shape of weather inputs (24 hours * 4 variables)
discriminator, disc_out = get_discriminator(demand_input, disc_operation_input, disc_weather_input)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Setting up the generator
discriminator.trainable = False
weather_input = Input(shape=(96,))  # input_x: shape of weather inputs (24 hours * 4 variables)
gen_operation_input = Input(shape=(5,))  # input_4: number of label categories (5)
noise_input = Input(shape=(101,))  # input_x: shape of latent noise (101)
generator, gen_out = get_generator(weather_input, gen_operation_input, noise_input)

# Setting up the GAN
x = generator([weather_input, gen_operation_input, noise_input])
gan_out = discriminator([x, disc_operation_input, disc_weather_input])
gan = Model([weather_input, gen_operation_input, noise_input, disc_operation_input, disc_weather_input], gan_out)
gan.summary()
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Empty array to store samples for experience replay, and documenting the loss progress
exp_replay = []
cntr = 0
shw = 0
perform_list = np.array([[0, 0, 0, 0, 0]])

# Training loop
for epoch in range(N_EPOCHS):
    cum_d_loss = 0.
    cum_g_loss = 0.

    for batch_idx in range(num_batches):
        # Soft targets with decay
        noise_flip = 0.1  # percentage of labels to flip
        noise_prop = np.maximum(noise_flip - ((noise_flip/(N_EPOCHS-1)) * 0.9 * epoch), 0)  # decay rate: 0.9
        # Train discriminator on real data
        images = generate_images(batch_idx, noise_prop)
        labels = y_operation[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        weather_data = z_weather[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        true_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0, high=noise_prop*2, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop * len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
        d_loss_true = discriminator.train_on_batch([images, labels, weather_data], true_labels)
        # Train generator with poisoned labels (felix culpa)
        anomaly = 1  # One sample with a poisoned label is enough to induce felix culpa
        labels = generate_real_labels(batch_idx, anomaly)
        noise_data = generate_noise(BATCH_SIZE, 101)
        random_labels = generate_random_labels(BATCH_SIZE)
        felix_culpa = np.random.normal(0.5, 0.05, size=(BATCH_SIZE, 1))
        g_loss_felix = gan.train_on_batch([weather_data, random_labels, noise_data, labels, weather_data], felix_culpa)
        # Train discriminator on generated data
        labels = y_operation[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        noise_data = generate_noise(BATCH_SIZE, 101)
        generated_images = generator.predict([weather_data, labels, noise_data])
        gene_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0, high=noise_prop*2, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop * len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
        d_loss_gene = discriminator.train_on_batch([generated_images, labels, weather_data], gene_labels)
        # Experience replay if batch size is smaller than total samples
        if BATCH_SIZE < n_train_samples:
            # Store a random point for experience replay
            r_idx = np.random.randint(BATCH_SIZE)
            exp_replay.append([generated_images[r_idx], labels[r_idx], weather_data[r_idx]])
            # If we have enough points, do experience replay
            if len(exp_replay) == BATCH_SIZE:
                generated_images = np.array([p[0] for p in exp_replay])
                replay_labels = np.array([p[1] for p in exp_replay])
                weather_data = np.array([p[2] for p in exp_replay])
                replay_correction_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0, high=0.1,
                                                                                        size=(BATCH_SIZE, 1))
                exp_d_loss = discriminator.train_on_batch([generated_images, replay_labels, weather_data], gene_labels)
                print(exp_d_loss)
                exp_replay = []
                break
        d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
        cum_d_loss += d_loss
        # Train generator
        random_labels = generate_random_labels(BATCH_SIZE)
        noise_data = generate_noise(BATCH_SIZE, 101)
        labels = y_operation[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        g_loss = gan.train_on_batch([weather_data, random_labels, noise_data, random_labels, weather_data],
                                    np.zeros((BATCH_SIZE, 1)))
        cum_g_loss += g_loss
        # Print performance
        print('\tEpoch: {}, Discriminator1 Loss: {}, Discriminator2 Loss: {}, Generator Loss: {}'.format(epoch + 1,
                                                                                                         d_loss_true,
                                                                                                         d_loss_gene,
                                                                                                         g_loss))
    perform_batch = np.array([[epoch + 1, d_loss_true, d_loss_gene, g_loss, g_loss_felix]])
    perform_list = np.concatenate((perform_list, perform_batch), axis=0)


# Save GAN and training performance
save_path = 'C:/Users/'
name_of_file = 'build9'
save_path = os.path.join(save_path, name_of_file+".mat")
scipy.io.savemat(save_path, {'cGAN_perf': perform_list})
generator.save('build9.h5')
