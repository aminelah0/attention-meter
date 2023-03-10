#Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from attention.img_proc.img_split import ImageCrop


def plot_multi_images(image_list: list[np.ndarray]) -> plt:
    '''Shows multiple images in 1 row'''
    fig, axs = plt.subplots(1, len(image_list), figsize=(20, 10))
    for i, image in enumerate(image_list):
        axs[i].imshow(image)
    return plt.show()


def plot_crops(crops: list[ImageCrop]):
    '''Shows all crops from original image as per how they were split along the axes'''
    n_columns = crops[-1].i_x + 1
    n_rows = crops[-1].i_y + 1
    fig = plt.figure(figsize=(20, 10))
    for i, crop in enumerate(crops):
        plt.subplot(n_columns, n_rows, i+1)
        plt.imshow(crop.image)
    return plt.show()
