import os

import numpy as np
import matplotlib.pyplot as plt


def cloud_to_rgb():

    color_ind = 0
    for file in os.listdir('C:/Users/ga25mal/OneDrive - TUM/2023_i3CE/data/instances/pipe/'):
    # for file in os.listdir('data/in/'):

        color = plt.cm.tab10(color_ind)
        color_rgb = [int(255 * c) for c in color[:3]]

        with open(f'C:/Users/ga25mal/OneDrive - TUM/2023_i3CE/data/instances/pipe/{file}', 'r') as f:
        # with open(f'data/in/{file}', 'r') as f:
            lines = f.readlines()
            # to np array
            lines = np.array([line.split() for line in lines])
            # drop columns
            lines = lines[:, :3]

            # write to file
            # with open(f'data/clean_clouds/{file}', 'w') as g:
            with open(f'data/clean_clouds_viz/{file}', 'w') as g:
                for line in lines:
                    g.write(f'{line[0]} {line[1]} {line[2]} {color_rgb[0]} {color_rgb[1]} {color_rgb[2]} \n')

        color_ind += 1
        if color_ind > 9:
            color_ind = 0

def obj_to_rgb():

    color_ind = 0
    for file in os.listdir('data/out/'):


if __name__ == '__main__':
    cloud_to_rgb()