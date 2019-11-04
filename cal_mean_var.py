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
mean = np.array([np.mean(weights[:,col_id][np.abs(weights[:,col_id])>1e-6]) for col_id in range(17*3)])
std = np.array([np.std(weights[:,col_id][np.abs(weights[:,col_id])>1e-6]) for col_id in range(17*3)])

np.save("mean_std.npy", {'mean':mean, 'std':std})

