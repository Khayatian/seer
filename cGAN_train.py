from tensorflow.keras.models import load_model
import numpy as np
import scipy.io
import os.path


# One-hot-encode values into Boolean format
def one_hot_encode(y, lbl):
    z = np.zeros((len(y), lbl))  # number of labels
    idx = np.arange(len(y))
    z[idx, y] = 1
    return z


# Generating latent noise
def generate_noise(n_samples, noise_dim):
    lat_noise = np.random.normal(0, 1, size=(n_samples, noise_dim))
    return lat_noise


# Load the trained model and the conditions
generator = load_model('CGAN_CiLr_build1a.h5')  # load saved generator
mat = scipy.io.loadmat('build1.mat')  # load Matlab data set
y_operation = mat['operation']  # load operation variables
z_weather = mat['weather']  # load weather variables


# Creating synthetic demands
num_samples = 50  # number of synthetic demand scenarios
data = np.ones((num_samples, y_operation.shape[0], 24, 6))  # Placeholder for synthetic demands
for smple in range(num_samples):
    noise_data = generate_noise(y_operation.shape[0], 96)  # generate latent noise
    #  Project synthetic samples for an entire year (365 samples)
    data[smple, :, :, :] = generator.predict([z_weather, y_operation, noise_data]).reshape((y_operation.shape[0], 24, 6))


# Save synthetic profiles
save_path = 'C:/Users/khfa/Documents/MATLAB/GANCityLearn/'
name_of_file = 'cGAN_pred_CiLr_build1a'
save_path = os.path.join(save_path, name_of_file+".mat")
scipy.io.savemat(save_path, {'data': data})
