import math


IMG_X = 32
# IMG_X = 64
# IMG_X = 40
# IMG_X = 50
# IMG_X = 128
# IMG_X = 256

IMG_Y = IMG_X
PIX_NUM = (IMG_X*IMG_Y)
PRJ_NUM = IMG_X
ANG_NUM = 180
BIN_NUM = (PRJ_NUM*ANG_NUM)
XRS = 0
DTC = 1
PI = math.pi
I_MIN = 0.0

CLOCKWISE = -1
USE_CAST = 0.49999999
PVB = 2.0    # pixel value of bone

MU_H2O = 1.0
LPP  = 1.0

m_num = 52
angle_num = 1

lam = 1e-1
p = 0.1
niter = 10

devide = 10

theta_list_construct=[179,178,177,176,175,0,1,2,3,4
                      ]

top_n_softmax = 10

temperature_list = [0.1, 0.5, 1.0, 2.0, 10, 100]

max_shots = 200
