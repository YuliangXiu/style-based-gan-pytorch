import scipy.io as sio 
import numpy as np 
import os 
from tqdm import tqdm
mat_dir = "./data/Meshes/PoseUnit/"

# weights = np.zeros((10200,17,3))
# ids = 0

# for _,_, mat_files in os.walk(mat_dir):
#     for mat_file in tqdm(mat_files):
#         if "mat" in mat_file:
#             weights[ids] = sio.loadmat(os.path.join(mat_dir, mat_file))['theta'].reshape(-1,3)
#             ids += 1

# np.save("theta.npy", weights)

weights = np.load("theta.npy").reshape(-1,17*3)
mean = np.mean(weights, axis=0)
std = np.std(weights, axis=0)
np.save("mean_std.npy", {'mean':mean, 'std':std})

