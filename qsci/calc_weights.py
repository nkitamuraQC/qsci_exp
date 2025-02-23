import numpy as np


class CalcWeights:
    def __init__(self, ny=20, max_y=10, nz=20, max_z=10):
        self.ny = ny
        self.max_y = max_y
        self.nz = nz
        self.max_z = max_z
        self.min_z = -max_z
        self.gen_y_space()
        self.gen_z_space()

    def gen_y_space(self):
        y = np.linspace(0, self.max_y, self.ny)
        self.delta_y = y[1] - y[0]
        return

    def gen_z_space(self):
        z = np.linspace(self.min_z, self.max_z, self.nz)
        self.delta_z = z[1] - z[0]
        return

    def get_weights(self, jy, jz):
        w = 1.0j / np.sqrt(np.pi * 2)
        w *= self.delta_y * self.delta_z * (jz * self.delta_z + self.min_z)
        w *= np.exp(-0.5 * (jz * self.delta_z + self.min_z) ** 2)
        return w

    def get_weights_k(self, jy, jz, k):
        w = 1.0j / np.sqrt(np.pi * 2)
        w *= self.delta_y * self.delta_z * (jz * self.delta_z + self.min_z)
        w *= np.exp(-0.5 * (jz * self.delta_z + self.min_z) ** 2)
        w *= (jy * self.delta_y) ** k
        return w
