import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

root_fall = "../mmfalldata/DS1_4falls.npy"

falls = np.load(root_fall)

def draw_frame(frame,ax):
    xd = []
    yd = []
    zd = []
    print(np.shape(frame))
    for x, y, z, _ in frame:
        xd.append(x)
        yd.append(y)
        zd.append(z)
    ax.scatter3D(xd,yd,zd, cmap="Blues")

if __name__ == '__main__':
    for item in falls:
        fig = plt.figure()
        ax = Axes3D(fig)
        for idx, frame in enumerate(item):
            draw_frame(frame, ax)
            if idx == 0:
                plt.show()
            else:
                plt.draw()