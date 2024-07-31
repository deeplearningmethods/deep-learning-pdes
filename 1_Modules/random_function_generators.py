from abc import ABC, abstractmethod
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftn, ifftn
from utils import fftfreq_multi, multi_to_flat


# Define the base class that all random function generators should implement
class RandomFunctionGenerator(ABC):
    def __init__(self):
        self.generator_name = None

    @abstractmethod
    def generate(self, batch_size, nr_spacediscr):
        pass

# Karhunen-Loeve expansion of a Gaussian random field with exponential decay
class KLSineCosExpansion1D(RandomFunctionGenerator):
    def __init__(self, params):
        super().__init__()
        self.var, self.nr_modes, self.decay_rate, self.offset = params
        self.generator_name = "Periodic KL with %s modes, var %s, decay_rate %s, offset %s" % (self.var, self.nr_modes, self.decay_rate, self.offset)

    def generate(self, batch_size, nr_spacediscr):
        normals = self.var * np.random.randn(batch_size, 2 * self.nr_modes + 1)
        factors = np.array(
            [1 / np.power((2 * np.pi * n) * (2 * np.pi * n) + self.offset, self.decay_rate) for n in range(1, self.nr_modes + 1)])

        coefficients_sin = normals[:, :self.nr_modes] * factors
        coefficients_cos = normals[:, self.nr_modes:2 * self.nr_modes] * factors
        coefficient_const = normals[:, 2 * self.nr_modes:] / np.power(self.offset, self.decay_rate)

        sin_basis_values = np.array(
            [[np.sqrt(2) * np.sin(2 * n * np.pi * i / nr_spacediscr) for i in range(nr_spacediscr)] for n in
             range(1, self.nr_modes + 1)])
        cos_basis_values = np.array(
            [[np.sqrt(2) * np.cos(2 * n * np.pi * i / nr_spacediscr) for i in range(nr_spacediscr)] for n in
             range(1, self.nr_modes + 1)])
        const_basis_values = np.array([[1 for i in range(nr_spacediscr)]])

        return np.matmul(coefficients_sin, sin_basis_values) + np.matmul(coefficients_cos, cos_basis_values) + np.matmul(coefficient_const, const_basis_values)


# Karhunen-Loeve expansion of a Gaussian random field with exponential decay based on IFFT
class RandnFourierSeriesGenerator(RandomFunctionGenerator):
    '''
        This creates a random function with the fourier coefficients being a Gaussian random field with exponential decay.
        Specifically, the fourier coefficients are of the form
        X_k = N(0, var * (offset + inner_decay * |k|^2)^(-decay_rate))
        where k is the wave number, var, offset, inner_decay and decay_rate are parameters.

    '''
    def __init__(self, params):
        super().__init__()
        self.var, self.decay_rate, self.offset, self.inner_decay, self.space_size, self.dim = params
        self.generator_name = f"Multidim FNO Style expantion with var {self.var}, decay_rate {self.decay_rate}, offset {self.offset}, inner_decay {self.inner_decay}, space_size {self.space_size}, dim {self.dim}"
        self.local_ifft = lambda x_hat: np.power(self.space_size, -self.dim / 2.) * ifftn(x_hat, axes=list(range(1, self.dim + 1)), norm="forward")

    def generate(self, batch_size, nr_spacediscr):
        '''
        :return: numpy array of shape (batch_size, nr_spacediscr ** self.dim)
        output_multi[i1, i2, ..., id] = u(x1, x2, ..., xd) where xi = i * space_size / nr_spacediscr for i in range(nr_spacediscr)
        '''

        mesh_step = self.space_size / nr_spacediscr
        fft_freqs = 2 * np.pi * fftfreq_multi(nr_spacediscr, self.dim, d=mesh_step)
        laplace_evs = np.sum(np.square(fft_freqs), axis=0)
        scale_factors = self.var * np.power(self.offset + self.inner_decay * laplace_evs, -self.decay_rate)
        random_coeff_real = np.random.randn(batch_size, *([nr_spacediscr] * self.dim))
        random_coeff_img = np.random.randn(batch_size, *([nr_spacediscr] * self.dim))
        random_coeff = random_coeff_real + 1j * random_coeff_img
        fft_coeffs = random_coeff * scale_factors

        output_multi = self.local_ifft(fft_coeffs).real

        # sqrt(2) * (abs(sigma). * ((my_const. * (1:N)').^2 + tau^2).^(-gamma/2));


        return multi_to_flat(output_multi, dim=self.dim)

class RandnFourierSeriesGeneratorStartControl(RandomFunctionGenerator):
    def __init__(self, params):
        super().__init__()
        self.var, self.decay_rate, self.offset, self.inner_decay, self.space_size, self.start_var, self.dim = params
        self.generator_name = f"Multidim FNO Style expantion with var {self.var}, decay_rate {self.decay_rate}, offset {self.offset}, inner_decay {self.inner_decay}, space_size {self.space_size}, start_var {self.start_var}, dim {self.dim}"
        self.local_ifft = lambda x_hat: np.power(self.space_size, -self.dim / 2.) * ifftn(x_hat, axes=list(range(1, self.dim + 1)), norm="forward")

    def generate(self, batch_size, nr_spacediscr):
        '''
        :return: numpy array of shape (batch_size, nr_spacediscr ** self.dim)
        output_multi[i1, i2, ..., id] = u(x1, x2, ..., xd) where xi = i * space_size / nr_spacediscr for i in range(nr_spacediscr)
        '''

        mesh_step = self.space_size / nr_spacediscr
        fft_freqs = 2 * np.pi * fftfreq_multi(nr_spacediscr, self.dim, d=mesh_step)
        laplace_evs = np.sum(np.square(fft_freqs), axis=0)
        scale_factors = self.var * np.power(self.offset + self.inner_decay * laplace_evs, -self.decay_rate)
        scale_factors[(0,) * self.dim] = self.start_var
        random_coeff_real = np.random.randn(batch_size, *([nr_spacediscr] * self.dim))
        random_coeff_img = np.random.randn(batch_size, *([nr_spacediscr] * self.dim))
        random_coeff = random_coeff_real + 1j * random_coeff_img
        fft_coeffs = random_coeff * scale_factors
        # fft_coeffs[(slice(None),) + (0,) * self.dim] = self.start_var * fft_coeffs[(slice(None),) + (0,) * self.dim]
        # fft_coeffs[(slice(None),) + (0,) * self.dim] = fft_coeffs[(slice(None),) + (0,) * self.dim] / 5

        output_multi = self.local_ifft(fft_coeffs).real

        return multi_to_flat(output_multi, dim=self.dim)




