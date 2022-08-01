"""
Script to create a gif out of images the rendezvous trajectory.
"""

import argparse
import time
import os
from PIL import Image
import sys


def gif_from_images(args, i_0=0, i_end=100):
    """
    Create a gif by combining images. The gif is stored as 'gif.gif' in the same directory where the images are.
    The images must be labeled as 'img#.png'.
    To crop the gif, check out https://ezgif.com/crop \n
    :param args: Namespace containing the arguments from the command line.
    :param i_0: # of the first image to be included in the gif.
    :param i_end: # of the last image to be included in the gif.
    :return: None
    """

    # Directory of the images:
    directory = args.path

    images = []
    missing = []
    available = []
    for i in range(i_0, i_end + 1):
        image_name = f'img{i}.png'
        try:
            images.append(Image.open(os.path.join(directory, image_name)))
            available.append(i)
        except FileNotFoundError:
            missing.append(i)

    # Save the gif if any images were found:
    if len(available) > 0:

        if len(missing) == 0:
            print('All images found.')
        else:
            print(f'{len(missing)} of {len(missing) + len(available)} images were not found.')

        out_path = os.path.join('plots', 'gif.gif')  # path of the output gif file.
        print(f'Saving gif of {len(available)} images in {out_path}')
        images[0].save(out_path, save_all=True, append_images=images[1:], duration=50, loop=0)

    else:
        print(f'No compatible images found in directory "{directory}"')
        print('Check if the names of the images and the directory are correct.')
        print('Exiting')
        sys.exit()

    return


def get_args():
    """
    Get the arguments when running the script from the command line.
    The `dir` argument specifies where the images are. It is always required.
    :return: Namespace containing the arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', type=str, default='', help='Path to the folder containing the images.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    start = time.perf_counter()
    arguments = get_args()
    # arguments.path = ''

    # Check that the path argument exists and is not a file:
    if os.path.isfile(arguments.path) or not os.path.exists(arguments.path):
        print(f'Incorrect --path argument "{arguments.path}"')
        print('Path must be an existing directory.\nExiting')
        sys.exit()

    gif_from_images(arguments, i_0=0, i_end=100)

    print(f'Finished on {time.ctime()}. ({round(time.perf_counter()-start, 2)} seconds)')
