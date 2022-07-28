"""
Learn how to create a compound object.
"""

import os
import numpy as np
from vpython import *
from utils.general import load_data
from utils.quaternions import quat_error, quat2rot
from utils.vpython_utils import create_scene, numpy2vec, create_chaser, create_target, create_koz

path = os.path.join('data', 'rdv_data5.pickle')
data = load_data(path)
# t = data['t']
rc = data['rc']  # chaser position [m]
qc = data['qc']  # chaser attitude
assert rc.shape[0] == 3

print('Computing rotations...', end=' ')

# print(qc[:, 0])
rot0 = quat2rot(qc[:, 0])
qc_axes = [numpy2vec(rot0[0])]
qc_angles = [rot0[1]]

for i in range(1, qc.shape[1]):
    qc_error = quat_error(qc[:, i-1], qc[:, i])
    axis, angle = quat2rot(qc_error)
    qc_axes.append(numpy2vec(axis))
    qc_angles.append(angle)
print('Done')

# Create a scene and the objects:
scene = create_scene(caption='')
chaser = create_chaser(rc0=rc[:, 0])
target = create_target()
koz = create_koz()
# scene.camera.follow(chaser)  # set the camera to follow the chaser

k_max = len(rc[0])
# Graph:
gd = graph(title='Test', xmin=0, xmax=k_max + 5, ymin=-50, ymax=10,
           align='left', ytitle='Position [m]', xtitle='Time [s]')
f1 = gcurve(color=color.red, label='x')
f2 = gcurve(color=color.green, label='y')
f3 = gcurve(color=color.blue, label='z')
f1.plot([0, chaser.pos.x])
f2.plot([0, chaser.pos.y])
f3.plot([0, chaser.pos.z])

# Main loop:
k = 1
while k < k_max:
    rate(30)
    chaser.pos = vector(rc[0, k], rc[1, k], rc[2, k])
    chaser.rotate(angle=qc_angles[k], axis=qc_axes[k], origin=chaser.pos)
    f1.plot([k, chaser.pos.x])
    f2.plot([k, chaser.pos.y])
    f3.plot([k, chaser.pos.z])
    k += 1
    # if k < len(rc[0])-1:
    #     k += 1
    # else:
    #     k = 0  # TODO: reset the rotation!
