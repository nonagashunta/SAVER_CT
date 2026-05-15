import math
from numexpr.necompiler import double
from tqdm import tqdm
import numpy as np
from constants import *
from structure import VEC2D, VEC2I


'''
Joseph's method
'''
# Calculate the coordinates of Xray and detector. Initialize the start and end point
# def calcCoordXrayAndDtc(point, radius, ratio):
#     # Size of the pixel
#     pix_sz = 1.0
#     # Size of the detector
#     dtc_sz = pix_sz * ratio
#     cur = 0
#     # Move the Xray form left to right in every detector, store the position of each ray
#     for dtc in range(PRJ_NUM):
#         for offset in np.arange(0.0, pix_sz, dtc_sz):
#             # Variable point stores the data of the central point of each start point of Xray and detector
#             point[cur, XRS].x =float(dtc + offset - 0.5*IMG_X + dtc_sz/2.0)
#             # radius = IMG_X / 2.0 if IMG_X<IMG_Y else IMG_Y / 2.0
#             # This makes the center of the image as the zero point
#             point[cur, XRS].y = -radius - 1.0
#             point[cur, DTC].x = point[cur, XRS].x
#             point[cur, DTC].y = - point[cur, XRS].y
#             cur += 1

def calcCoordXrayAndDtc(point, radius, ratio):
    pix_sz = 1.0
    dtc_sz = pix_sz * ratio
    cur = 0

    # PRJ_NUM 本の検出器が 画像幅にちょうど収まるように配置
    for dtc in range(PRJ_NUM):
        for offset in np.arange(0.0, pix_sz, dtc_sz):
            # 0.5 * PRJ_NUM に変更
            point[cur, XRS].x = float(dtc + offset - 0.5 * PRJ_NUM + dtc_sz / 2.0)
            point[cur, XRS].y = -radius - 1.0
            point[cur, DTC].x = point[cur, XRS].x
            point[cur, DTC].y = - point[cur, XRS].y
            cur += 1


# Rotate the ray
def rot(vec, theta, dir):
    rot_vec = VEC2D()
    rot_vec.x = math.cos(theta)*vec.x - dir*math.sin(theta)*vec.y
    rot_vec.y = dir * math.sin(theta) * vec.x + math.cos(theta) * vec.y
    return rot_vec

# Transfer the ray
def trans(vec, b):
    trans_vec = VEC2D()
    trans_vec.x = vec.x + b.x
    trans_vec.y = vec.y + b.y
    return trans_vec

def calc_radius(a, b):
    return math.sqrt((a - IMG_X / 2.0)**2+ (b - IMG_Y / 2.0)**2)

def calcOneRay1(img, point, theta, radius):
    '''
    Input the data of image
    '''
    pix_sz = 1.0
    coord = VEC2D(IMG_X / 2.0, IMG_Y / 2.0)
    ax = 0.0
    idxs = VEC2I()
    start = VEC2D()
    end = VEC2D()
    cxr = VEC2D()

    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    start = rot(point[XRS], theta, CLOCKWISE)
    start = trans(start, coord)
    end = rot(point[DTC], theta, CLOCKWISE)
    end = trans(end, coord)

    dir = 1 if start.y<end.y else -1
    # from 0 to IMG_Y - 1
    idxs.y = 0 if start.y<end.y else (IMG_Y - 1)
    slope = sin_th/cos_th
    while idxs.y>=0 and idxs.y < IMG_Y:
        cxr.y = idxs.y + pix_sz/2.0
        cxr.x = start.x + dir * slope * abs(cxr.y - start.y)
        if USE_CAST == None:
            idxs.x = int(round(cxr.x - 1.0))
        else:
            tmp = cxr.x - USE_CAST
            idxs.x = int(tmp - 1.0) if tmp<0.0 else int(tmp)

        pix = idxs.y * IMG_X + idxs.x
        dist = idxs.x - cxr.x + 1.0
        #print(f"Iteration: idxs.y={idxs.y}, cxr.x={cxr.x}, slope={slope}, dir={dir}")
        if calc_radius(idxs.x, idxs.y) <= radius:
            ax += img[pix] * (0.5 + dist)
        if calc_radius(idxs.x + 1, idxs.y) <= radius:
            ax += img[pix + 1] * (0.5 - dist)
        idxs.y += dir
    return ax / abs(cos_th)


def calcOneRay2(img, point, theta, radius):
    pix_sz = 1.0
    coord = VEC2D(IMG_X / 2.0, IMG_Y / 2.0)
    ax = 0.0
    offset = 0.0
    idxs = VEC2I()
    start = VEC2D()
    end = VEC2D()
    cxr = VEC2D()

    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    start = rot(point[XRS], theta, CLOCKWISE)
    start = trans(start, coord)
    end = rot(point[DTC], theta, CLOCKWISE)
    end = trans(end, coord)

    dir = 1 if start.x < end.x else -1
    idxs.x = 0 if start.x < end.x else (IMG_X - 1)
    slope = cos_th / sin_th
    while idxs.x >= 0 and idxs.x < IMG_X:
        cxr.x = idxs.x + pix_sz / 2.0
        cxr.y = start.y + dir * slope * abs(cxr.x - start.x)
        if USE_CAST == None:
            idxs.y = int(round(cxr.y - 1.0))
        else:
            tmp = cxr.y - USE_CAST
            idxs.y = int(tmp - 1.0) if tmp < 0.0 else int(tmp)

        pix = idxs.y*IMG_X + idxs.x
        dist = idxs.y - cxr.y + 1.0
        if 0 <= (idxs.y) < IMG_Y and calc_radius(idxs.x, idxs.y) <= radius:
            ax += img[pix] * (0.5 + dist)
        if 0 <= (idxs.y + 1) < IMG_Y and calc_radius(idxs.x, idxs.y + 1) <= radius:
            ax += img[pix + IMG_X] * (0.5 - dist)

        idxs.x += dir
    return ax / abs(sin_th)

def forward_projection(prj, img, ratio, mode):
    dtc_num = int(PRJ_NUM / ratio)
    radius = IMG_X / 2.0 if IMG_X<IMG_Y else IMG_Y / 2.0
    bone_coeff = 2.0 / PVB
    cols = 2
    for bin in range(BIN_NUM):
        prj[bin] = 0.0

    point = np.zeros((dtc_num, cols), dtype=object)
    for i in range(dtc_num):
        for j in range(cols):
            point[i, j] = VEC2D(x=0, y=0)

    calcCoordXrayAndDtc(point, radius, ratio)
    bin = 0
    for angle in range(ANG_NUM):
        theta = angle * PI / ANG_NUM
        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        cur = 0
        for dtc in range(PRJ_NUM):
            offset = 0.0
            for offset in np.arange(0.0, 1.0, ratio):
                if abs(sin_th) <= abs(cos_th):
                    b = calcOneRay1(img, point[cur], theta, radius)
                else:
                    b = calcOneRay2(img, point[cur], theta, radius)
                if mode == "NORMAL":
                    prj[bin] += b
                else:
                    prj[bin] += (b * bone_coeff * MU_H2O * LPP * ratio)
                cur += 1
            bin += 1

def updateImage1(img, pvalue, point, theta, radius):
    pix_sz = 1.0
    coord = VEC2D(IMG_X / 2.0, IMG_Y / 2.0)
    ax = 0.0
    idxs = VEC2I()
    start = VEC2D()
    end = VEC2D()
    cxr = VEC2D()

    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    start = rot(point[XRS], theta, CLOCKWISE)
    start = trans(start, coord)
    end = rot(point[DTC], theta, CLOCKWISE)
    end = trans(end, coord)

    dir = 1 if start.y < end.y else -1
    idxs.y = 0 if start.y < end.y else (IMG_Y - 1)
    slope = sin_th / cos_th

    while idxs.y >= 0 and idxs.y < IMG_Y:
        # coordinate of Xray
        cxr.y = idxs.y + pix_sz / 2.0
        cxr.x = start.x + dir * slope * abs(cxr.y - start.y)
        if USE_CAST == None:
            idxs.x = int(round(cxr.x - 1.0))
        else:
            tmp = cxr.x - USE_CAST
            idxs.x = int(tmp - 1.0) if tmp < 0.0 else int(tmp)

        pix = idxs.y * IMG_X + idxs.x
        dist = idxs.x - cxr.x + 1.0
        if calc_radius(idxs.x, idxs.y) <= radius:
            img[pix] += pvalue * ((0.5 + dist) / abs(cos_th))
        if calc_radius(idxs.x + 1, idxs.y) <= radius:
            img[pix + 1] += pvalue * ((0.5 - dist) / abs(cos_th))
        idxs.y += dir


def updateImage2(img, pvalue, point, theta, radius):
    '''
    Input img as an initial input, pvalue as the projection value
    '''
    pix_sz = 1.0
    coord = VEC2D(IMG_X / 2.0, IMG_Y / 2.0)
    ax = 0.0
    idxs = VEC2I()
    start = VEC2D()
    end = VEC2D()
    cxr = VEC2D()

    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    start = rot(point[XRS], theta, CLOCKWISE)
    start = trans(start, coord)
    end = rot(point[DTC], theta, CLOCKWISE)
    end = trans(end, coord)

    dir = 1 if start.x < end.x else -1
    idxs.x = 0 if start.x < end.x else (IMG_X - 1)
    slope = cos_th / sin_th
    while idxs.x >= 0 and idxs.x < IMG_X:
        cxr.x = idxs.x + pix_sz / 2.0
        cxr.y = start.y + dir * slope * abs(cxr.x - start.x)
        if USE_CAST == None:
            idxs.y = int(round(cxr.y - 1.0))
        else:
            tmp = cxr.y - USE_CAST
            idxs.y = int(tmp - 1.0) if tmp < 0.0 else int(tmp)

        pix = idxs.y*IMG_X + idxs.x
        dist = idxs.y - cxr.y + 1.0
        if 0 <= (idxs.y) < IMG_Y and  calc_radius(idxs.x, idxs.y) <= radius:
            img[pix] += pvalue * ((0.5 + dist) / abs(sin_th))
        if 0 <= (idxs.y + 1) < IMG_Y and  calc_radius(idxs.x, idxs.y + 1) <= radius:
            img[pix + IMG_X] += pvalue * ((0.5 - dist) / abs(sin_th))

        idxs.x += dir

def back_projection(img, prj):
    ratio = 0.25
    dtc_num = int(PRJ_NUM / ratio) #1024
    radius = IMG_X / 2.0 if IMG_X < IMG_Y else IMG_Y / 2.0
    bone_coeff = 2.0 / PVB
    offset = 0.0
    cols = 2
    for pix in range(PIX_NUM):
        img[pix] = I_MIN

    point = np.zeros((dtc_num, cols), dtype=object)
    for i in range(dtc_num):
        for j in range(cols):
            point[i, j] = VEC2D(x=0, y=0)


    calcCoordXrayAndDtc(point, radius, ratio)
    bin = 0
    for angle in range(ANG_NUM):
        theta = angle * PI / ANG_NUM
        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        cur = 0
        for dtc in range(PRJ_NUM):
            for offset in np.arange(0.0, 1.0, ratio):
                if abs(sin_th) <= abs(cos_th):
                    updateImage1(img, prj[bin], point[cur], theta, radius)
                else:
                    updateImage2(img, prj[bin], point[cur], theta, radius)
                cur += 1
            bin += 1
    pix = 0
    for i in range(IMG_Y):
        for j in range(IMG_X):
            if (calc_radius(j, i) > radius):
                img[pix] = I_MIN
            else:
                img[pix] *= double(ratio/ANG_NUM)
            pix += 1


# Reconstructure the image with SIRT
def joseph(alpha, prj_real, x_real, mode, iterations):
    ratio = 0.25
    x = np.zeros(PIX_NUM)  # Initialize the image estimate x with zeros
    prj_calc = np.zeros_like(prj_real)  # Array to store the calculated forward projection
    img = np.zeros(PIX_NUM, dtype=np.float32)  # Independent array for back projection updates

    for iteration in tqdm(range(iterations), desc="Iterations"):
        # Compute the forward projection of the current estimate x into prj_calc
        forward_projection(prj_calc, x, ratio, mode)
        # Calculate the residual: real projection - calculated projection
        b0 = prj_real - prj_calc
        # Reset img to zero and perform back projection
        print("b0:",np.linalg.norm(b0))
        img.fill(0.0)
        back_projection(img, b0)
        # Update x with the back-projected residual
        x += alpha * img
        # Reset prj_calc to zero for the next iteration
        prj_calc.fill(0.0)
        # Print progress every 10 iterations
        if np.linalg.norm(b0) < 1e-6:
            print("Converged!")
            break
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: max(x)={np.max(x)}, error={np.linalg.norm(x - x_real)}")
        print("img:",np.linalg.norm(img))
        if np.linalg.norm(img) < 1e-6:
            break

    print("Final result:", x)  # Print the final result
    print("error:", np.linalg.norm(x - x_real))
    return x  # Return the reconstructed image

