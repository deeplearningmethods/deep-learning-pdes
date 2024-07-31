import torch
import numpy as np
from ode_methods import explicit_euler, second_order_lirk, second_order_lirk_fem, ODE, SemiLinearODE, SemiLinearFEMODE, strang_splitting
from scipy.fft import fft, ifft, fftfreq, fftn, ifftn
from random_function_generators import RandnFourierSeriesGenerator
import matplotlib.pyplot as plt
from utils import numpy_to_torch, torch_to_numpy
import time
from PDE_operations import fdm_laplace_operator_periodic, first_order_diff_matrix_trans


# Operations for discretizations for periodic burgers equation

def x_values_burgers(nr_spacediscr, space_size, dim=1, boundary=False):
    if boundary:
        return np.linspace(0, space_size, nr_spacediscr + 1)
    else:
        return np.linspace(0, space_size, nr_spacediscr + 1)[:-1]

def reduce_dimension_burgers(values, space_resolution_step, dim=1):
    return values[:, ::space_resolution_step]

def get_higher_nr_spacediscr_burgers(nr_spacediscr, space_resolution_step):
    return space_resolution_step * nr_spacediscr

def create_boundary_values_burgers(function_values):
    left_values = function_values[:, 0].reshape(-1, 1)
    output_array = np.concatenate((function_values, left_values), axis=1)
    return output_array




#######################################
#      FDM methods
#######################################



def burger_fdm_euler_np(initial_values, T, laplace_factor, space_size, nr_timesteps, diff_version = 2, conservative = True):
    # Computations done in numpy
    _, nr_spacediscr = initial_values.shape
    mesh_step = space_size / nr_spacediscr

    diff_matrix_trans = first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version)

    if conservative:
        nonlin = lambda u: - 0.5 * np.matmul(u * u, diff_matrix_trans)
    else:
        nonlin = lambda u: - u * (np.matmul(u, diff_matrix_trans))

    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)

    def dynamic(u):
        return np.matmul(u, operator) + nonlin(u)

    ode = ODE(ode_dynamic=dynamic)

    end_values = explicit_euler(T, ode, initial_values, nr_timesteps)

    return end_values


def burger_fdm_euler(initial_values, T, laplace_factor, space_size, nr_timesteps, diff_version = 2, conservative = True):
    # Computations done in pytorch
    """
        :param initial_values: [batchsize, nr_spacediscr] np.array
        :return: [batchsize, nr_spacediscr] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_spacediscr = initial_values.shape
    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    diff_matrix_trans_torch = numpy_to_torch(first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version), device)

    if conservative:
        nonlin = lambda u: - 0.5 * torch.matmul(u * u, diff_matrix_trans_torch)
    else:
        nonlin = lambda u: - u * (torch.matmul(u, diff_matrix_trans_torch))


    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)
    operator_torch = numpy_to_torch(operator, device)

    def dynamic(u):
        return torch.matmul(u, operator_torch) + nonlin(u)

    ode = ODE(ode_dynamic=dynamic)
    end_values = explicit_euler(T, ode, initial_values_torch, nr_timesteps)
    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values


def burger_fdm_lirk_np(initial_values, T, laplace_factor, space_size, nr_timesteps, v = 2, conservative = True):
    diff_version = v
    _, nr_spacediscr = initial_values.shape

    diff_matrix_trans = first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version)

    if conservative:
        nonlin = lambda u: - 0.5 * np.matmul(u * u, diff_matrix_trans)
    else:
        nonlin = lambda u: - u * (np.matmul(u, diff_matrix_trans))

    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)

    semilinear_ode = SemiLinearODE(operator, nonlin)

    end_values = second_order_lirk(T, semilinear_ode, initial_values, nr_timesteps, [0.5, 0.5])

    return end_values


def burger_fdm_lirk(initial_values, T, laplace_factor, space_size, nr_timesteps, diff_version = 2, conservative = True, params=[0.5, 0.5]):
    # Computations done in pytorch
    """
        :param initial_values: [batchsize, nr_spacediscr] np.array
        :return: [batchsize, nr_spacediscr] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_spacediscr = initial_values.shape
    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    diff_matrix_trans_torch = numpy_to_torch(first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version), device)

    if conservative:
        nonlin = lambda u: - 0.5 * torch.matmul(u * u, diff_matrix_trans_torch)
    else:
        nonlin = lambda u: - u * (torch.matmul(u, diff_matrix_trans_torch))

    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)
    operator_torch = numpy_to_torch(operator, device)

    semilinear_ode = SemiLinearODE(operator_torch, nonlin)

    end_values = second_order_lirk(T, semilinear_ode, initial_values_torch, nr_timesteps, params)
    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values

def burger_fdm_strang_splitting_exponential(initial_values, T, laplace_factor, space_size, nr_timesteps, diff_version = 2, conservative = True, params=[0.5, 0.5], implicit_first=True):
    """
        :param initial_values: [batchsize, nr_spacediscr] np.array
        :return: [batchsize, nr_spacediscr] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_spacediscr = initial_values.shape
    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    diff_matrix_trans_torch = numpy_to_torch(first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version), device)
    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)
    operator_torch = numpy_to_torch(operator, device)

    timestep = float(T)/nr_timesteps
    implicit_timestep = timestep/2. if implicit_first else timestep
    explicit_timestep = timestep if implicit_first else timestep/2.

    def diffusion_step(u):
        return torch.matmul(u, torch.matrix_exp(implicit_timestep * operator_torch))

    def nonlin_step(u):
        temp1 = torch.exp(-explicit_timestep * torch.matmul(u, diff_matrix_trans_torch)) * u
        # diff_matrix_trans_torch_expanded = diff_matrix_trans_torch.expand(batch_size, -1, -1)
        # matrices = diff_matrix_trans_torch_expanded * u.unsqueeze(1)
        # linear_flows = torch.empty_like(matrices)
        # for i in range(batch_size):
        #     linear_flows[i] = torch.matrix_exp(-explicit_timestep*matrices[i])
        # temp2 = torch.matmul(u.unsqueeze(1), linear_flows)
        # temp2 = temp2.squeeze(1)
        # return 0.5 * (temp1 + temp2)
        return  temp1

    if implicit_first:
        end_values = strang_splitting(diffusion_step, nonlin_step, initial_values_torch, nr_timesteps)
    else:
        end_values = strang_splitting(nonlin_step, diffusion_step, initial_values_torch, nr_timesteps)

    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values


def burger_fdm_strang_splitting_lax_wendroff(initial_values, T, laplace_factor, space_size, nr_timesteps, implicit_first=True):
    """
        :param initial_values: [batchsize, nr_spacediscr] np.array
        :return: [batchsize, nr_spacediscr] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_spacediscr = initial_values.shape
    mesh_step = space_size / nr_spacediscr
    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    diff_quot_symmetric = np.array([0, 1] + [0 for _ in range(nr_spacediscr - 3)] + [-1])
    diff_matrix_symmetric = numpy_to_torch(np.stack([np.roll(diff_quot_symmetric, n) for n in range(nr_spacediscr)]), device).t()
    diff_quot_left = np.array([1] + [0 for _ in range(nr_spacediscr - 2)] + [-1])
    diff_matrix_left = numpy_to_torch(np.stack([np.roll(diff_quot_left, n) for n in range(nr_spacediscr)]), device).t()
    diff_quot_right = np.array([-1, 1] + [0 for _ in range(nr_spacediscr - 2)])
    diff_matrix_right = numpy_to_torch(np.stack([np.roll(diff_quot_right, n) for n in range(nr_spacediscr)]), device).t()

    average_left = 0.5 * np.array([1] + [0 for _ in range(nr_spacediscr - 2)] + [1])
    average_matrix_left = numpy_to_torch(np.stack([np.roll(average_left, n) for n in range(nr_spacediscr)]), device).t()
    average_right = 0.5 * np.array([1, 1] + [0 for _ in range(nr_spacediscr - 2)])
    average_matrix_right = numpy_to_torch(np.stack([np.roll(average_right, n) for n in range(nr_spacediscr)]), device).t()

    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)
    operator_torch = numpy_to_torch(operator, device)

    timestep = float(T) / nr_timesteps
    implicit_timestep = timestep / 2. if implicit_first else timestep
    nonlin_timestep = timestep if implicit_first else timestep / 2.

    def diffusion_step(u):
        return torch.matmul(u, torch.matrix_exp(implicit_timestep * operator_torch))

    nonlin = lambda u: -0.5 * u * u
    nonlin_derivative = lambda u: -u

    def nonlin_step(u):
        first_order_term = nonlin_timestep / mesh_step / 2. * torch.matmul(nonlin(u), diff_matrix_symmetric)
        second_order_term_1 = nonlin_derivative(torch.matmul(u, average_matrix_right)) * torch.matmul(nonlin(u), diff_matrix_right)
        second_order_term_2 = nonlin_derivative(torch.matmul(u, average_matrix_left)) * torch.matmul(nonlin(u), diff_matrix_left)
        second_oder_term = nonlin_timestep**2 / mesh_step**2 / 2. * (second_order_term_1 - second_order_term_2)
        return u + first_order_term + second_oder_term

    if implicit_first:
        end_values = strang_splitting(diffusion_step, nonlin_step, initial_values_torch, nr_timesteps)
    else:
        end_values = strang_splitting(nonlin_step, diffusion_step, initial_values_torch, nr_timesteps)

    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values


def burger_fdm_strang_splitting_first_order(initial_values, T, laplace_factor, space_size, nr_timesteps, diff_version = 2, implicit_first=True):
    """
        :param initial_values: [batchsize, nr_spacediscr] np.array
        :return: [batchsize, nr_spacediscr] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_spacediscr = initial_values.shape
    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    diff_matrix_trans_torch = numpy_to_torch(first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version), device)
    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1)
    operator_torch = numpy_to_torch(operator, device)

    timestep = float(T)/nr_timesteps
    implicit_timestep = timestep/2. if implicit_first else timestep
    explicit_timestep = timestep if implicit_first else timestep/2.

    def diffusion_step(u):
        return torch.matmul(u, torch.matrix_exp(implicit_timestep * operator_torch))

    def nonlin_step(u):
        return torch.exp(-explicit_timestep * torch.matmul(u, diff_matrix_trans_torch)) * u

    if implicit_first:
        end_values = strang_splitting(diffusion_step, nonlin_step, initial_values_torch, nr_timesteps)
    else:
        end_values = strang_splitting(nonlin_step, diffusion_step, initial_values_torch, nr_timesteps)

    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values



#######################################
#      FEM methods
#######################################


def burger_fem_lirk_np(initial_values, T, laplace_factor, space_size, nr_timesteps, params=[0.5, 0.5]):

    _, nr_spacediscr = initial_values.shape
    mesh_step = space_size / nr_spacediscr

    stiff_row = laplace_factor / mesh_step * np.array([-2, 1] + [0 for _ in range(nr_spacediscr - 3)] + [1])
    stiff_matrix = np.stack([np.roll(stiff_row, n) for n in range(nr_spacediscr)])

    mass_row = mesh_step / 6 * np.array([4, 1] + [0 for _ in range(nr_spacediscr - 3)] + [1])
    mass_matrix = np.stack([np.roll(mass_row, n) for n in range(nr_spacediscr)])

    nonlin_fem_tensor = np.zeros((nr_spacediscr, nr_spacediscr, nr_spacediscr))
    lower_index = np.arange(nr_spacediscr)
    higher_index = np.arange(1, nr_spacediscr+1)%nr_spacediscr

    nonlin_fem_tensor[higher_index, lower_index, lower_index] = 1./3.
    nonlin_fem_tensor[lower_index, higher_index, higher_index] = -1./3.
    nonlin_fem_tensor[higher_index, higher_index, lower_index] = 1./6.
    nonlin_fem_tensor[higher_index, lower_index, higher_index] = 1./6.
    nonlin_fem_tensor[lower_index, higher_index, lower_index] = -1./6.
    nonlin_fem_tensor[lower_index, lower_index, higher_index] = -1./6.

    nonlin = lambda u: - np.einsum('bi,bij-> bj', u, np.einsum('bk,ikj', u, nonlin_fem_tensor))

    fem_ode = SemiLinearFEMODE(mass_matrix, stiff_matrix, nonlin)

    end_values = second_order_lirk_fem(T, fem_ode, initial_values, nr_timesteps, params)

    return end_values


def burger_fem_lirk(initial_values, T, laplace_factor, space_size, nr_timesteps, params=[0.5, 0.5]):
    # Computations done in pytorch
    """
        :param initial_values: [batchsize, nr_spacediscr] np.array
        :return: [batchsize, nr_spacediscr] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_spacediscr = initial_values.shape
    mesh_step = space_size / nr_spacediscr
    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values


    stiff_row = laplace_factor / mesh_step * np.array([-2, 1] + [0 for _ in range(nr_spacediscr - 3)] + [1])
    stiff_matrix = np.stack([np.roll(stiff_row, n) for n in range(nr_spacediscr)])
    stiff_matrix_torch = numpy_to_torch(stiff_matrix, device)

    mass_row = mesh_step / 6 * np.array([4, 1] + [0 for _ in range(nr_spacediscr - 3)] + [1])
    mass_matrix = np.stack([np.roll(mass_row, n) for n in range(nr_spacediscr)])
    mass_matrix_torch = numpy_to_torch(mass_matrix, device)

    nonlin_fem_tensor = np.zeros((nr_spacediscr, nr_spacediscr, nr_spacediscr))
    lower_index = np.arange(nr_spacediscr)
    higher_index = np.arange(1, nr_spacediscr+1)%nr_spacediscr

    nonlin_fem_tensor[higher_index, lower_index, lower_index] = 1./3.
    nonlin_fem_tensor[lower_index, higher_index, higher_index] = -1./3.
    nonlin_fem_tensor[higher_index, higher_index, lower_index] = 1./6.
    nonlin_fem_tensor[higher_index, lower_index, higher_index] = 1./6.
    nonlin_fem_tensor[lower_index, higher_index, lower_index] = -1./6.
    nonlin_fem_tensor[lower_index, lower_index, higher_index] = -1./6.
    nonlin_fem_tensor_torch = numpy_to_torch(nonlin_fem_tensor, device)

    nonlin = lambda u: - torch.einsum('bi,bij-> bj', u, torch.einsum('bk,ikj', u, nonlin_fem_tensor_torch))

    fem_ode = SemiLinearFEMODE(mass_matrix_torch, stiff_matrix_torch, nonlin)

    end_values = second_order_lirk_fem(T, fem_ode, initial_values_torch, nr_timesteps, params)
    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values



#######################################
#      Spectral methods
#######################################

# Not yet implemented in pytorch
def burgers_spectral_statesp_euler(initial_values, T, laplace_factor, space_size, nr_timesteps, conservative = True):
    _, nr_modes = initial_values.shape
    mesh_step = space_size / nr_modes

    der_ev =  1j * 2 * np.pi * fftfreq(nr_modes, d=mesh_step)
    local_fft = lambda x : np.sqrt(space_size) * fft(x, norm="forward")
    local_ifft = lambda x : 1 / np.sqrt(space_size) * ifft(x, norm="forward")

    def dynamic(u):
        linear_contrib = laplace_factor * local_ifft(der_ev * der_ev * local_fft(u)).real
        if conservative:
            nonlin_contrib = - 0.5 * local_ifft(der_ev * local_fft(u * u)).real
        else:
            nonlin_contrib = -local_ifft(der_ev * local_fft(u)).real * u
        return linear_contrib + nonlin_contrib

    ode = ODE(ode_dynamic=dynamic)

    end_values = explicit_euler(T, ode, initial_values, nr_timesteps)

    return end_values

# Not yet implemented in pytorch
def burgers_spectral_freqsp_euler(initial_values, T, laplace_factor, space_size, nr_timesteps, conservative = True):
    _, nr_modes = initial_values.shape
    mesh_step = space_size / nr_modes

    der_ev =  1j * 2 * np.pi * fftfreq(nr_modes, d=mesh_step)
    local_fft = lambda x : np.sqrt(space_size) * fft(x, norm="forward")
    local_ifft = lambda x : 1 / np.sqrt(space_size) * ifft(x, norm="forward")

    initial_modes = local_fft(initial_values)

    def dynamic(v):
        linear_contrib = laplace_factor * der_ev * der_ev * v
        if conservative:
            nonlin_contrib = - 0.5 * der_ev * local_fft(local_ifft(v).real * local_ifft(v).real)
        else:
            nonlin_contrib = - local_fft(local_ifft(der_ev * v).real * local_ifft(v).real)
        return linear_contrib + nonlin_contrib

    ode = ODE(ode_dynamic=dynamic)
    end_modes = explicit_euler(T, ode, initial_modes, nr_timesteps)
    end_values = local_ifft(end_modes).real
    return end_values


# Statespace does not work with LIRK (because the operator involves the Fourier transform)
def burgers_spectral_freqsp_lirk_np(initial_values, T, laplace_factor, space_size, nr_timesteps, conservative = True):
    _, nr_modes = initial_values.shape
    mesh_step = space_size / nr_modes

    der_ev =  1j * 2 * np.pi * fftfreq(nr_modes, d=mesh_step)
    local_fft = lambda x : np.sqrt(space_size) * fft(x, norm="forward")
    local_ifft = lambda x : 1 / np.sqrt(space_size) * ifft(x, norm="forward")

    initial_modes = local_fft(initial_values)

    if conservative:
        nonlin = lambda v: - 0.5 * der_ev * local_fft(local_ifft(v).real * local_ifft(v).real)
    else:
        nonlin = lambda v: - local_fft(local_ifft(der_ev * v).real * local_ifft(v).real)

    operator = np.diag(laplace_factor * der_ev * der_ev)

    semilinear_ode = SemiLinearODE(operator, nonlin)

    end_modes = second_order_lirk(T, semilinear_ode, initial_modes, nr_timesteps, [0.5, 0.5])
    end_values = local_ifft(end_modes).real

    return end_values

def burgers_spectral_freqsp_lirk(initial_values, T, laplace_factor, space_size, nr_timesteps, conservative = True):
    # Computations done in pytorch
    """
        :param initial_values: [batchsize, nr_spacediscr] np.array
        :return: [batchsize, nr_spacediscr] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_modes = initial_values.shape
    mesh_step = space_size / nr_modes
    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values
    initial_values_torch = initial_values_torch.to(torch.complex64)

    der_ev = (1j * 2 * np.pi * torch.fft.fftfreq(nr_modes, d=mesh_step)).to(device)
    local_fft = lambda x : torch.sqrt(torch.tensor(space_size)) * torch.fft.fft(x)
    local_ifft = lambda x : torch.fft.ifft(x) / torch.sqrt(torch.tensor(space_size))

    initial_modes = local_fft(initial_values_torch)

    # Define the nonlinearity
    if conservative:
        def nonlin(v):
            return - 0.5 * der_ev * local_fft(local_ifft(v).real * local_ifft(v).real)
    else:
        def nonlin(v):
            return - local_fft(local_ifft(der_ev * v).real * local_ifft(v).real)


    operator = torch.diag(laplace_factor * der_ev * der_ev).to(device)

    semilinear_ode = SemiLinearODE(operator, nonlin)

    end_modes = second_order_lirk(T, semilinear_ode, initial_modes, nr_timesteps, [0.5, 0.5])
    end_values = local_ifft(end_modes).real
    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values


if __name__ == "__main__":
    from training_samples_generators import get_data
    from PDE_operations import reduce_dimension_periodic, get_higher_nr_spacediscr_periodic

    reduce_dimension = reduce_dimension_periodic
    get_higher_nr_spacediscr = get_higher_nr_spacediscr_periodic

    #######################################
    # Test case for periodic Burgers PDE in 1d
    T = 2.
    space_size = 2 * np.pi
    laplace_factor = 0.1

    # initial value
    var = 1000
    decay_rate = 3.
    offset = np.power(var, 1 / decay_rate)
    inner_decay = 2.
    initial_value_generator = RandnFourierSeriesGenerator([var, decay_rate, offset, inner_decay, space_size, 1])

    nr_spacediscr = 128

    # Ref solution parameters
    ref_space_resolution_step = 4
    ref_nr_timesteps = 200
    batch_size = 100
    reference_algorithm = lambda initial_values, nr_timesteps: burgers_spectral_freqsp_lirk(initial_values, T, laplace_factor, space_size, nr_timesteps)

    ref_nr_spacediscr = get_higher_nr_spacediscr(nr_spacediscr, ref_space_resolution_step)

    print("Generating reference_solutions samples")
    _, _, initial_values, reference_solution = (
        get_data(
            initial_value_generator, reference_algorithm,
            batch_size, ref_nr_spacediscr, ref_nr_timesteps,
            reduce_dimension, ref_space_resolution_step, 'test',
        ))

    space_grid = x_values_burgers(nr_spacediscr, space_size)

    # Prepare plot
    fig_plot, ax_plot = plt.subplots(1, 1)
    fig_errors, ax_errors = plt.subplots(1, 1, figsize=(10, 6))

    l2_error = lambda u: np.linalg.norm(u - reference_solution)/nr_spacediscr

    # Run all methods, stop time, and plot results
    start = 1
    end = 8
    nr_timesteps = [1, 2, 3, 4, 5, 6] + [i for i in range(7, 20, 2)] + [i for i in range(21, 50, 3)] + [i for i in range(54, 100, 10)] #[2**i for i in range(start, end)]


    print("burger_fdm_euler:")
    start = time.perf_counter()
    burger_fdm_euler_values = [burger_fdm_euler(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
    print("  Time in pytorch: ", time.perf_counter() - start)
    start = time.perf_counter()
    burger_fdm_euler_np_values = [burger_fdm_euler_np(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
    print("  Time in numpy: ", time.perf_counter() - start)
    print("  Difference between numpy and pytorch implementation: ", np.linalg.norm(burger_fdm_euler_values[-1] - burger_fdm_euler_np_values[-1]))
    for i in range(len(nr_timesteps)):
        if np.max(np.abs(burger_fdm_euler_values[i] - reference_solution)) < 0.5:
            ax_plot.plot(space_grid, burger_fdm_euler_values[i][0], label="burger_fdm_euler_" + str(nr_timesteps[i]))
    # errors_euler = [np.linalg.norm(burger_fdm_euler_values[i] - reference_solution) for i in range(len(nr_timesteps))]
    # ax_errors.scatter(nr_timesteps, errors_euler, label="burger_fdm_euler")

    print("burger_fdm_lirk:")
    for v in [2]:
        for conservative in [False, True]:
            print(f"v = {v}, conservative = {conservative}")
            start = time.perf_counter()
            burger_fdm_lirk_values = [burger_fdm_lirk(initial_values, T, laplace_factor, space_size, nr_timesteps, v, conservative) for nr_timesteps in nr_timesteps]
            print("  Time in pytorch: ", time.perf_counter() - start)
            start = time.perf_counter()
            burger_fdm_lirk_np_values = [burger_fdm_lirk_np(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
            print("  Time in numpy: ", time.perf_counter() - start)
            print("  Difference between numpy and pytorch implementation: ", np.linalg.norm(burger_fdm_lirk_values[-1] - burger_fdm_lirk_np_values[-1]))
            for i in range(len(nr_timesteps)):
                if np.max(np.abs(burger_fdm_lirk_values[i] - reference_solution)) < 0.5:
                    ax_plot.plot(space_grid, burger_fdm_lirk_values[i][0], label="burger_fdm_lirk_" + str(nr_timesteps[i]))
            errors_lirk = [l2_error(burger_fdm_lirk_values[i]) for i in range(len(nr_timesteps))]
            ax_errors.scatter(nr_timesteps, errors_lirk, label=f"burger_fdm_lirk_{v}_{conservative}")
            # Estimate rate
            log_errors_lirk = np.log(np.array(errors_lirk))
            log_nr_timesteps = np.log(np.array(nr_timesteps))
            slope = (log_errors_lirk[1:] - log_errors_lirk[:-1]) / (log_nr_timesteps[1:] - log_nr_timesteps[:-1])
            print("Estimated rate in time: ", np.mean(slope))

    # print("burger_fem_lirk:")
    # start = time.perf_counter()
    # burger_fem_lirk_values = [burger_fem_lirk(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
    # print("  Time in pytorch: ", time.perf_counter() - start)
    # start = time.perf_counter()
    # burger_fem_lirk_np_values = [burger_fem_lirk_np(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
    # print("  Time in numpy: ", time.perf_counter() - start)
    # print("  Difference between numpy and pytorch implementation: ", np.linalg.norm(burger_fem_lirk_values[-1] - burger_fem_lirk_np_values[-1]))
    # for i in range(len(nr_timesteps)):
    #     ax_plot.plot(space_grid, burger_fem_lirk_values[i][0], label="burger_fem_lirk_" + str(nr_timesteps[i]))
    # errors_fem_lirk = [np.linalg.norm(burger_fem_lirk_values[i] - reference_solution) for i in range(len(nr_timesteps))]
    # ax_errors.scatter(nr_timesteps, errors_fem_lirk, label="burger_fem_lirk")

    print("burger_fdm_strang_splitting_exponential:")
    for implicit_first in [True, False]:
        start = time.perf_counter()
        burger_fdm_strang_splitting_values = [burger_fdm_strang_splitting_exponential(initial_values, T, laplace_factor, space_size, nr_timesteps, implicit_first=implicit_first) for nr_timesteps in nr_timesteps]
        print("  Time in pytorch: ", time.perf_counter() - start)
        for i in range(len(nr_timesteps)):
            if np.max(np.abs(burger_fdm_strang_splitting_values[i] - reference_solution)) < 0.5:
                ax_plot.plot(space_grid, burger_fdm_strang_splitting_values[i][0], label="burger_fdm_strang_splitting_exponential_" + str(nr_timesteps[i]))
        errors_strang_splitting = [l2_error(burger_fdm_strang_splitting_values[i]) for i in range(len(nr_timesteps))]
        ax_errors.scatter(nr_timesteps, errors_strang_splitting, label=f"burger_fdm_strang_splitting_exponential_{implicit_first}")
        # Estimate rate
        log_errors_strang_splitting = np.log(np.array(errors_strang_splitting))
        log_nr_timesteps = np.log(np.array(nr_timesteps))
        slope = (log_errors_strang_splitting[1:] - log_errors_strang_splitting[:-1]) / (log_nr_timesteps[1:] - log_nr_timesteps[:-1])
        print(f"Estimated rate for implicit first = {implicit_first} in time: ", np.mean(slope))

    print("burger_fdm_strang_splitting_lax_wendroff:")
    for implicit_first in [True, False]:
        start = time.perf_counter()
        burger_fdm_strang_splitting_lax_wendroff_values = [burger_fdm_strang_splitting_lax_wendroff(initial_values, T, laplace_factor, space_size, nr_timesteps, implicit_first=implicit_first) for nr_timesteps in nr_timesteps]
        print("  Time in pytorch: ", time.perf_counter() - start)
        for i in range(len(nr_timesteps)):
            if np.max(np.abs(burger_fdm_strang_splitting_lax_wendroff_values[i] - reference_solution)) < 0.5:
                ax_plot.plot(space_grid, burger_fdm_strang_splitting_lax_wendroff_values[i][0], label="burger_fdm_strang_splitting_lax_wendroff_" + str(nr_timesteps[i]))
        errors_strang_splitting_lax_wendroff = [l2_error(burger_fdm_strang_splitting_lax_wendroff_values[i]) for i in range(len(nr_timesteps))]
        ax_errors.scatter(nr_timesteps, errors_strang_splitting_lax_wendroff, label=f"burger_fdm_strang_splitting_lax_wendroff_{implicit_first}")
        # Estimate rate
        log_errors_strang_splitting_lax_wendroff = np.log(np.array(errors_strang_splitting_lax_wendroff))
        log_nr_timesteps = np.log(np.array(nr_timesteps))
        slope = (log_errors_strang_splitting_lax_wendroff[1:] - log_errors_strang_splitting_lax_wendroff[:-1]) / (log_nr_timesteps[1:] - log_nr_timesteps[:-1])
        print(f"Estimated rate for implicit first = {implicit_first} in time: ", np.mean(slope))

    burger_spectral_statesp_euler_values = [burgers_spectral_statesp_euler(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
    for i in range(len(nr_timesteps)):
        if np.max(np.abs(burger_spectral_statesp_euler_values[i] - reference_solution)) < 0.5:
            ax_plot.plot(space_grid, burger_spectral_statesp_euler_values[i][0], label="burger_spectral_statesp_euler_" + str(nr_timesteps[i]))
    errors_spectral_statesp_euler = [l2_error(burger_spectral_statesp_euler_values[i]) for i in range(len(nr_timesteps))]
    # ax_errors.scatter(nr_timesteps, errors_spectral_statesp_euler, label="burger_spectral_statesp_euler")

    burger_spectral_freqsp_euler_values = [burgers_spectral_freqsp_euler(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
    for i in range(len(nr_timesteps)):
        if np.max(np.abs(burger_spectral_freqsp_euler_values[i] - reference_solution)) < 0.5:
            ax_plot.plot(space_grid, burger_spectral_freqsp_euler_values[i][0], label="burger_spectral_freqsp_euler_" + str(nr_timesteps[i]))
    errors_spectral_freqsp_euler = [l2_error(burger_spectral_freqsp_euler_values[i]) for i in range(len(nr_timesteps))]
    # ax_errors.scatter(nr_timesteps, errors_spectral_freqsp_euler, label="burger_spectral_freqsp_euler")

    print("burger_spectral_freqsp_lirk:")
    start = time.perf_counter()
    burger_spectral_freqsp_lirk_values = [burgers_spectral_freqsp_lirk(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
    print("  Time in pytorch: ", time.perf_counter() - start)
    start = time.perf_counter()
    burger_spectral_freqsp_lirk_np_values = [burgers_spectral_freqsp_lirk_np(initial_values, T, laplace_factor, space_size, nr_timesteps) for nr_timesteps in nr_timesteps]
    print("  Time in numpy: ", time.perf_counter() - start)
    print("  Difference between numpy and pytorch implementation: ", np.linalg.norm(burger_spectral_freqsp_lirk_values[-1] - burger_spectral_freqsp_lirk_np_values[-1]))
    for i in range(len(nr_timesteps)):
        if np.max(np.abs(burger_spectral_freqsp_lirk_values[i] - reference_solution)) < 0.5:
            ax_plot.plot(space_grid, burger_spectral_freqsp_lirk_values[i][0], label="burger_spectral_freqsp_lirk_" + str(nr_timesteps[i]))
    errors_spectral_freqsp_lirk = [l2_error(burger_spectral_freqsp_lirk_values[i]) for i in range(len(nr_timesteps))]
    ax_errors.scatter(nr_timesteps, errors_spectral_freqsp_lirk, label="burger_spectral_freqsp_lirk")
    # Estimate rate
    log_errors_spectral_freqsp_lirk = np.log(np.array(errors_spectral_freqsp_lirk))
    log_nr_timesteps = np.log(np.array(nr_timesteps))
    slope = (log_errors_spectral_freqsp_lirk[1:] - log_errors_spectral_freqsp_lirk[:-1]) / (log_nr_timesteps[1:] - log_nr_timesteps[:-1])
    print("Estimated rate in time: ", np.mean(slope))

    ax_plot.plot(space_grid, initial_values[0], label="initial value")
    ax_plot.plot(space_grid, reference_solution[0], label="reference solution")
    ax_plot.legend()
    fig_plot.show()

    ax_errors.set_yscale('log')
    ax_errors.set_xscale('log')
    # Save legend outside of the plot
    ax_errors.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig_errors.tight_layout()
    fig_errors.savefig("Z_burgers_convergence_tests.pdf")
    fig_errors.show()


