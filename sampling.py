

from joseph import *


def calculate_projection_of_sample_point(img, x0, y0, theta, ratio, mode):
    '''
    Function: Calculate the projection from the original image by the ray going through point(x0, y0)
              with angle theta
    '''
    # Size of the pixel
    pix_sz = 1.0
    # Size of the detector
    dtc_sz = pix_sz * ratio
    dtc_num = int(pix_sz / dtc_sz)
    center = VEC2D(IMG_X / 2.0, IMG_Y / 2.0)
    sample_shift = VEC2D(x0 - center.x, y0 - center.y) # Centralize
    sample_unrot = rot(sample_shift, -theta, CLOCKWISE)
    centralX = sample_unrot.x

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    bone_coeff = 2.0 / PVB


    radius = IMG_X / 2
    prj = 0
    # r = x0 * cos_th + y0 * sin_th

    for offset in range(dtc_num):
        # Variable point stores the data of the central point of each start point of Xray and detector
        effectiveX= float(centralX + offset * dtc_sz + dtc_sz / 2.0)
        # radius = IMG_X / 2.0 if IMG_X<IMG_Y else IMG_Y / 2.0
        # This makes the center of the image as the zero point
        start_unrot = VEC2D(effectiveX, -radius - 1.0)
        end_unrot = VEC2D(effectiveX, radius + 1.0)


        ray = [start_unrot, end_unrot]

        if abs(sin_th) <= abs(cos_th):
            b = calcOneRay1(img, ray, theta, radius)
        else:
            b = calcOneRay2(img, ray, theta, radius)
        if mode == "NORMAL":
            prj += b
        else:
            prj += (b * bone_coeff * MU_H2O * LPP * ratio)

    return prj

def calculate_projection_of_single_ray(img, r, theta, ratio, mode):
    # Size of the pixel
    pix_sz = 1.0
    # Size of the detector
    dtc_sz = pix_sz * ratio
    dtc_num = int(pix_sz / dtc_sz)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    bone_coeff = 2.0 / PVB

    radius = IMG_X / 2
    prj = 0

    for offset in range(dtc_num):
        # Variable point stores the data of the central point of each start point of Xray and detector
        detector_position = r - radius + offset * dtc_sz + dtc_sz / 2.0
        # radius = IMG_X / 2.0 if IMG_X<IMG_Y else IMG_Y / 2.0
        # This makes the center of the image as the zero point
        start_unrot = VEC2D(detector_position, -radius - 1.0)
        end_unrot = VEC2D(detector_position, radius + 1.0)

        ray = [start_unrot, end_unrot]

        if abs(sin_th) <= abs(cos_th):
            b = calcOneRay1(img, ray, theta, radius)
        else:
            b = calcOneRay2(img, ray, theta, radius)
        if mode == "NORMAL":
            prj += b
        else:
            prj += (b * bone_coeff * MU_H2O * LPP * ratio)

    return prj

