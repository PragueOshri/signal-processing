import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import math

# implemant a function that simulate the firing rate of interneurons in teh positioning system of the cockroach
'''
Inputs:
    1. r0 - baseline firing rate
    2. n - number of interneurons
    3. alpha_i - vector of preferred angle alpha of each neuron
    4. alpha - real angle
Output: 
    1. alpha_hat - the prediction of the neurons usign population vactors
'''


def calc_prediction_angle(r0, n, list_angles, real_angle):
    list_angles = np.array(list_angles)

    # calculate the firing rate for each neuron
    list_ri = r0 * np.cos(real_angle - list_angles)
    list_ri[list_ri < 0] = 0

    # convert list of angels to diration vectors
    v = []
    for angle in list_angles:
        vi = [np.cos(angle), np.sin(angle)]
        v.append(vi)

    # since we are using cos function it's blocked, therefore the r_max = r0
    pred = np.dot(((list_ri - r0) / r0), v)

    # convert the radian to angle
    pred = math.atan2(pred[1], pred[0]) * 180 / np.pi % 360
    if pred < 0:
        pred = 360 + pred
    return pred
