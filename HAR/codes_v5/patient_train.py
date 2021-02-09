import numpy as np


if __name__ == '__main__':
    filepath = '../patient/bin/x_falling1_0.npy'
    filepath2 = '../patient/bin/x_falling2_0.npy'
    test = np.load(filepath)
    test2 = np.load(filepath2)
    print(np.shape(test),np.shape(test2))