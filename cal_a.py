from sampling import calculate_projection_of_single_ray
import numpy as np
from const import *
from tqdm import tqdm

# Calculate the list of possible choices of rays
# Each vector 'a' represents a choice of ray

a_list = []

# Calculate all vector 'a's
for j in tqdm(range(ANG_NUM)):
    theta = j * PI / ANG_NUM
    for t in range(PRJ_NUM):
        r = t
        a = np.zeros((PIX_NUM, 1))

        for i in range(PIX_NUM):
            e = np.zeros(PIX_NUM)
            e[i] = 1
            prj = calculate_projection_of_single_ray(e, r, theta, ratio=0.25, mode="NULL")
            a[i, 0] = prj
        
        a_list.append(a)

np.save("a_list_ratio0.25_mode_NULL.npy", a_list)