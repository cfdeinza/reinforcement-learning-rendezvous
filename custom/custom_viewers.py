"""
Nor sure if I need this file. My intention was to make custom viewers instead of using the classical_control viewer. 
However, it might be simpler to use matplotlib directly.

Written by C. F. De Inza Niemeijer.
"""

import numpy as np
import pyglet
from gym import error


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )

    pass


def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs
    )


class AttitudeViewer:
    def __init__(self, width, height, display=None):
        display = get_display(display)
        self.width = width
        self.height = height
        self.window = get_window(width=width, height=height, display=display)
        self.isopen = True

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
        pass



