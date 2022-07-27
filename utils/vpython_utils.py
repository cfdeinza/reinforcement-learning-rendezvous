"""
Utility functions for the vpython animations.
"""

from vpython import *
import numpy as np


def create_scene(title='Title', caption='Caption'):
    """
    Create a new scene.
    :param title: title on the canvas
    :param caption: caption on the canvas
    :return: the scene object
    """

    # Create a scene from a canvas:
    myscene = canvas(
        title=title + '\n',
        caption=caption,
        background=color.gray(0.99),
        align='left',
    )

    # Change the camera and light options:
    myscene.select()
    myscene.up = vector(0, 0, 1)
    myscene.forward = vector(-0.5, 1, -0.75)
    light1_dir = vector(-0.2, -1, 0.5)
    myscene.lights[0].direction = light1_dir
    myscene.lights[1].direction = -light1_dir

    return myscene


def create_chaser(rc0: np.ndarray):
    """
    Create a compound object to represent the chaser.\n
    :param rc0: Initial position of the chaser
    :return: vpython compound object
    """
    assert rc0.shape == (3,)

    cl, ch, cw = (0.5, 1.0, 0.5)  # chaser dimensions
    chaser_pos = vector(rc0[0], rc0[1], rc0[2])
    chaser_body = box(
        pos=chaser_pos,
        size=vector(cl, ch, cw),
        color=color.orange,
        # texture=textures.stucco,  # cannot have individual textures for compound object
    )

    ctl, cth, ctw = (cl / 2, 0.2, cw / 2)  # chaser tip dimensions
    chaser_tip = box(
        pos=chaser_pos + vector(0, (ch + cth) / 2, 0),
        size=vector(ctl, cth, ctw),
        color=color.gray(0.5),
        # opacity=0.4,
    )

    chaser = compound(
        [chaser_body, chaser_tip],
        origin=chaser_pos,  # set the position of the compound body (default is the center of the bounding box)
        # texture={'file': 'sat.png'},
        # shininess=1.0,
    )
    return chaser


def create_target(koz_radius=None):
    """
    Create a compound object to represent the target (and the entry corridor).\n
    :param koz_radius: radius of the keep-out zone [m]
    :return: vpython compound object
    """

    if koz_radius is None:
        koz_radius = 5
        print(f'Radius of keep-out zone is undefined. Using default ({koz_radius} m)')
    tl, th, tw = (1, 1, 1)
    target_body = box(
        pos=vector(0, 0, 0),
        size=vector(tl, th, tw),
        color=color.yellow,
        shininiess=0,
    )

    ttl, tth, ttw = (0.25, 0.2, 0.25)  # target tip dimensions
    target_tip = box(
        pos=vector(0, -(th + tth) / 2, 0),
        size=vector(ttl, tth, ttw),
        color=color.gray(0.5),
        shininess=0,
        # opacity=0.4,
    )

    tal, tah, taw = (1.25 * tl, 0.05, 0.05)  # target axle dimensions
    target_axle = box(
        pos=vector(0, 0, 0),
        size=vector(tal, tah, taw),
        color=color.gray(0.5),
    )

    pl, ph, pw = (tl * 2, taw, th * 0.8)
    panel1 = box(
        pos=vector((tal + pl) / 2, 0, 0),
        size=vector(pl, ph, pw),
        color=vector(0.19, 0.73, 0.96),
        shininess=1,
        # opacity=0.4,
        # texture={'file': 'solar.png'},
        # emissive=True,
    )

    panel2 = panel1.clone(pos=-panel1.pos)

    # Entry corridor:
    cone_half_angle = np.radians(30)
    cone_length = koz_radius * np.cos(cone_half_angle)
    cone_axis = vector(0, 1, 0) * cone_length
    corridor = cone(
        pos=-cone_axis,  # located at the center of the base of the cone
        axis=cone_axis,  # points from pos to the tip of the cone
        radius=cone_length * np.tan(cone_half_angle),  # radius of the base of the cone
        color=color.green,
        opacity=0.2,
        shininess=0,
    )

    target = compound(
        [target_body, target_axle, panel1, panel2, target_tip, corridor],
        origin=vector(0, 0, 0),
        # texture={'file': 'solar3.png'}
    )

    return target


def create_koz(koz_radius=None):
    """
    Create a sphere to represent the keep-out zone.\n
    :param koz_radius: radius of the keep-out zone.
    :return: vpython sphere() object
    """

    if koz_radius is None:
        koz_radius = 5
        print(f'Radius of keep-out zone is undefined. Using default ({koz_radius} m)')

    koz = sphere(
        pos=vector(0, 0, 0),
        radius=koz_radius,
        opacity=0.1,
        color=color.red,
        shininess=0,
    )

    return koz


def play_pause_button(b):
    """
    Callback function to play/pause the animation when the user presses a button. Usage example:
    `button(text="Play", pos=myscene.title_anchor, bind=play_pause_button)`\n
    :param b: vpython button object
    :return: None
    """
    global running
    running = not running
    if running:
        b.text = "Pause"
    else:
        b.text = " Play "
    return


def numpy2vec(arr: np.ndarray):
    """
    Convert a (3,) numpy array into a vpython vector.
    :param arr: numpy array
    :return: vpython vector object
    """

    if arr is None:
        return None
    else:
        assert arr.shape == (3,)
        return vector(arr[0], arr[1], arr[2])
