import torch
import numpy as np
from utils import numpy_to_torch, torch_to_numpy, flat_to_multi, multi_to_flat, fftfreq_multi, pad_zeros_in_middle, remove_middle
from ode_methods import second_order_lirk_diag_operator, SemiLinearODEDiagOperator, SemiLinearODE, second_order_lirk, strang_splitting_semilinear
from random_function_generators import RandnFourierSeriesGeneratorStartControl
from PDE_operations import fdm_laplace_operator_periodic

def periodic_semilinear_pde_fdm_lirk(initial_values, T, laplace_factor, nonlin, space_size, nr_timesteps, dim=1, params=(0.5, 0.5)):
    """
    :param initial_values: [batchsize, nr_spacediscr**dim] np.array
    :return: [batchsize, nr_spacediscr**dim] np.array
    """
    if dim > 2:
        raise NotImplementedError("Only dim=1,2 is implemented at the moment")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_gridpoints = initial_values.shape
    nr_spacediscr = int(np.power(nr_gridpoints, 1/dim) + 0.5)
    assert nr_gridpoints == nr_spacediscr**dim, "Initial values cannot be transformed to a square grid"

    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=dim)
    operator_torch = numpy_to_torch(operator, device)
    semilinear_ode = SemiLinearODE(operator_torch, nonlin)

    end_values = second_order_lirk(T, semilinear_ode, initial_values_torch, nr_timesteps, params)

    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values


def periodic_semilinear_pde_fdm_strang_splitting(initial_values, T, laplace_factor, nonlin, space_size, nr_timesteps, dim=1, params=(0.5, 0.5), implicit_first=True):
    """
    :param initial_values: [batchsize, nr_spacediscr**dim] np.array
    :return: [batchsize, nr_spacediscr**dim] np.array
    """
    if dim > 2:
        raise NotImplementedError("Only dim=1,2 is implemented at the moment")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_gridpoints = initial_values.shape
    nr_spacediscr = int(np.power(nr_gridpoints, 1/dim) + 0.5)
    assert nr_gridpoints == nr_spacediscr**dim, f"Initial values cannot be transformed to a square grid\n nr_gridpoints: {nr_gridpoints}, nr_spacediscr**dim:{nr_spacediscr**dim}"

    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    operator = laplace_factor * fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=dim)
    operator_torch = numpy_to_torch(operator, device)
    semilinear_ode = SemiLinearODE(operator_torch, nonlin)

    end_values = strang_splitting_semilinear(T, semilinear_ode, initial_values_torch, nr_timesteps, params, implicit_first)

    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values


def periodic_semilinear_pde_spectral_lirk(initial_values, T, laplace_factor, nonlin, space_size, nr_timesteps, dim=1, params=(0.5, 0.5)):
    """
    :param initial_values: [batchsize, nr_spacediscr**dim] np.array
    :return: [batchsize, nr_spacediscr**dim] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_gridpoints = initial_values.shape
    nr_spacediscr = int(np.power(nr_gridpoints, 1/dim) + 0.5)
    assert nr_gridpoints == nr_spacediscr**dim, "Initial values cannot be transformed to a square grid"

    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    local_fft = lambda x: np.power(space_size, dim/2.) * torch.fft.fftn(x, dim=list(range(1, dim+1)), norm="forward")
    local_ifft = lambda x_hat: np.power(space_size, -dim/2.) * torch.fft.ifftn(x_hat, dim=list(range(1, dim+1)), norm="forward")

    initial_modes = multi_to_flat(local_fft(flat_to_multi(initial_values_torch, dim)))

    local_nonlin = lambda x_hat :  multi_to_flat(local_fft(nonlin(local_ifft(flat_to_multi(x_hat, dim)).real)))

    mesh_step = space_size / nr_spacediscr
    fft_freqs = 2 * np.pi * fftfreq_multi(nr_spacediscr, dim, d=mesh_step)
    second_der_evs = - np.square(fft_freqs)
    operator_evs_multi = laplace_factor * np.sum(second_der_evs, axis=0)  # Should this be done in pytorch?
    operator_evs_flat = operator_evs_multi.flatten()
    operator_evs_flat = numpy_to_torch(operator_evs_flat, device)

    semilinear_ode = SemiLinearODEDiagOperator(operator_evs_flat, local_nonlin)
    end_modes = second_order_lirk_diag_operator(T, semilinear_ode, initial_modes, nr_timesteps, params)
    end_values = multi_to_flat(local_ifft(flat_to_multi(end_modes, dim=dim)).real)

    return torch_to_numpy(end_values) if isinstance(initial_values, np.ndarray) else end_values


def periodic_semilinear_pde_spectral_lirk_rough(initial_values, T, laplace_factor, nonlin, space_size, nr_timesteps, nr_modes, dim=1, params=(0.5, 0.5)):
    """
    :param initial_values: [batchsize, nr_spacediscr**dim] np.array
    :return: [batchsize, nr_spacediscr**dim] np.array
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_gridpoints = initial_values.shape
    nr_spacediscr = int(np.power(nr_gridpoints, 1/dim) + 0.5)
    assert nr_gridpoints == nr_spacediscr**dim, "Initial values cannot be transformed to a square grid"
    assert nr_modes <= nr_spacediscr, 'Number of modes must be less than or equal to the space discretization points'

    initial_values_torch = numpy_to_torch(initial_values, device) if isinstance(initial_values, np.ndarray) else initial_values

    local_fft = lambda x : np.power(space_size, dim/2.) * torch.fft.fftn(x, dim=list(range(1, dim+1)), norm="forward")
    local_ifft = lambda x_hat : np.power(space_size, -dim/2.) * torch.fft.ifftn(x_hat, dim=list(range(1, dim+1)), norm="forward")
    local_nonlin = lambda x_hat:  multi_to_flat(local_fft(nonlin(local_ifft(flat_to_multi(x_hat, dim)).real)))

    initial_values_multi = flat_to_multi(initial_values_torch, dim)
    initial_modes_multi = local_fft(initial_values_multi)
    initial_modes_multi_rough = remove_middle(initial_modes_multi, nr_modes)
    initial_modes_flat_rough = multi_to_flat(initial_modes_multi_rough, dim=dim)

    mesh_step = space_size / nr_modes
    fft_freqs = 2 * np.pi * fftfreq_multi(nr_modes, dim, d=mesh_step)
    second_der_evs = - np.square(fft_freqs)
    operator_evs_multi = laplace_factor * np.sum(second_der_evs, axis=0)
    operator_evs_flat = operator_evs_multi.flatten()
    operator_evs_flat = numpy_to_torch(operator_evs_flat, device)

    semilinear_ode = SemiLinearODEDiagOperator(operator_evs_flat, local_nonlin)
    end_modes_multi_rough = flat_to_multi(second_order_lirk_diag_operator(T, semilinear_ode, initial_modes_flat_rough, nr_timesteps, params), dim=dim)
    end_modes_multi = pad_zeros_in_middle(end_modes_multi_rough, nr_spacediscr)

    end_values_multi = local_ifft(end_modes_multi).real
    end_values_flat = multi_to_flat(end_values_multi, dim=dim)

    return torch_to_numpy(end_values_flat) if isinstance(initial_values, np.ndarray) else end_values_flat


if __name__ == "__main__":

    # Test periodic_semilinear_pde_spectral_lirk_rough
    import time
    import matplotlib.pyplot as plt
    from random_function_generators import RandnFourierSeriesGenerator
    from utils import numpy_to_torch, torch_to_numpy, flat_to_multi, multi_to_flat, fftfreq_multi
    from ode_methods import second_order_lirk_diag_operator, SemiLinearODEDiagOperator

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)


    # Set parameters
    batch_size = 10
    nr_spacediscr = 300
    T = 1.
    space_size = 1.
    laplace_factor = 0.002
    # nonlin = lambda x: - x * x * x + x
    nonlin = lambda x: - torch.sin(x)
    dim = 1

    # Create initial values
    var = 5000
    decay_rate = 2
    offset = np.power(var, 1 / decay_rate)
    inner_decay = 1.
    start_var = 0.2
    initial_value_generator = RandnFourierSeriesGeneratorStartControl([var, decay_rate, offset, inner_decay, space_size, start_var, dim])
    initial_values = initial_value_generator.generate(batch_size, nr_spacediscr)

    # Compute reference solutions
    nr_ref_timesteps = 1000
    ref_end_values = periodic_semilinear_pde_spectral_lirk(initial_values, T, laplace_factor, nonlin, space_size, nr_ref_timesteps, dim)

    # Prepare plots
    fig_errors, ax_errors = plt.subplots(1, 1)
    ax_errors.set_xscale('log')
    ax_errors.set_yscale('log')
    ax_errors.set_xlabel('Number of timesteps')
    ax_errors.set_ylabel('Error')

    nr_timesteps = [2**i for i in range(2, 8)]
    log_nr_timesteps = np.log(np.array(nr_timesteps))


    # Test periodic_semilinear_pde_fdm_lirk
    print("Testing periodic_semilinear_pde_fdm_lirk")
    params = [0.5, 0.5]
    # Compute all errors, scatter plot them
    errors = [np.linalg.norm(periodic_semilinear_pde_fdm_lirk(initial_values, T, laplace_factor, nonlin, space_size, nr_timestep, dim, params) - ref_end_values) for nr_timestep in nr_timesteps]
    ax_errors.scatter(nr_timesteps, errors, label="periodic_semilinear_pde_spectral_lirk")
    # Estimate rate
    log_errors = np.log(np.array(errors))
    slope = (log_errors[-1] - log_errors[0]) / (log_nr_timesteps[-1] - log_nr_timesteps[0])
    print("Estimated rate in time: ", slope)

    # Test periodic_semilinear_pde_fdm_strang_splitting
    print("Testing periodic_semilinear_pde_fdm_strang_splitting")
    params = [0.5, 0.5]
    # Compute all errors, scatter plot them
    for implicit_first in [True, False]:
        errors = [np.linalg.norm(periodic_semilinear_pde_fdm_strang_splitting(initial_values, T, laplace_factor, nonlin, space_size, nr_timestep, dim, params, implicit_first) - ref_end_values) for nr_timestep in nr_timesteps]
        ax_errors.scatter(nr_timesteps, errors, label=f"fdm_strang_splitting_implicit_first={implicit_first}")
        # Estimate rate
        log_errors = np.log(np.array(errors))
        slope = (log_errors[-1] - log_errors[0]) / (log_nr_timesteps[-1] - log_nr_timesteps[0])
        print(f"Estimated rate in time for implicit_first={implicit_first}: ", slope)

    # Test periodic_semilinear_pde_spectral_lirk
    print("Testing periodic_semilinear_pde_spectral_lirk")
    params = [0.5, 0.5]
    # Compute all errors, scatter plot them
    errors = [np.linalg.norm(periodic_semilinear_pde_spectral_lirk(initial_values, T, laplace_factor, nonlin, space_size, nr_timestep, dim, params) - ref_end_values) for nr_timestep in nr_timesteps]
    ax_errors.scatter(nr_timesteps, errors, label="periodic_semilinear_pde_spectral_lirk")
    # Estimate rate
    log_errors = np.log(np.array(errors))
    slope = (log_errors[-1] - log_errors[0]) / (log_nr_timesteps[-1] - log_nr_timesteps[0])
    print("Estimated rate in time: ", slope)

    # Test periodic_semilinear_pde_spectral_lirk_rough
    print("Testing periodic_semilinear_pde_spectral_lirk_rough")
    params = [0.5, 0.5]
    nr_modes = 16
    # Compute all errors, scatter plot them
    errors = [np.linalg.norm(periodic_semilinear_pde_spectral_lirk_rough(initial_values, T, laplace_factor, nonlin, space_size, nr_timestep, nr_modes, dim, params) - ref_end_values) for nr_timestep in nr_timesteps]
    ax_errors.scatter(nr_timesteps, errors, label="periodic_semilinear_pde_spectral_lirk_rough")
    # Estimate rate
    log_errors = np.log(np.array(errors))
    slope = (log_errors[-1] - log_errors[0]) / (log_nr_timesteps[-1] - log_nr_timesteps[0])
    print("Estimated rate in time: ", slope)

    fig_errors.legend()
    fig_errors.show()



    ############################################################
    if False:
        # Leave this away for the moment

        # Compare rough and standard spectral lirk
        nr_timesteps = 100

        # Run periodic_semilinear_pde_spectral_lirk
        start_time = time.time()
        end_values = periodic_semilinear_pde_spectral_lirk(initial_values, T, laplace_factor, nonlin, space_size, nr_timesteps, dim)
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time}")
        print(f"initial_values.shape: {initial_values.shape}")
        print(f"end_values.shape: {end_values.shape}")




        # Run periodic_semilinear_pde_spectral_lirk_rough
        nr_modes = 16
        start_time = time.time()
        end_values_rough = periodic_semilinear_pde_spectral_lirk_rough(initial_values, T, laplace_factor, nonlin,
                                                                       space_size, nr_timesteps, nr_modes, dim)
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time}")
        print(f"initial_values.shape: {initial_values.shape}")
        print(f"end_values_rough.shape: {end_values_rough.shape}")

        # Plot results if dim = 1
        nr_of_trajectories = 1
        if dim == 1:
            for i in range(nr_of_trajectories):
                plt.plot(np.linspace(0, space_size, nr_spacediscr), initial_values[i], label=f"Initial values {i}")
                plt.plot(np.linspace(0, space_size, nr_spacediscr), end_values[i], label=f"End values {i}")
                plt.plot(np.linspace(0, space_size, nr_spacediscr), end_values_rough[i], label=f"End values rough {i}")
            plt.legend()
            plt.show()

            # Calculate error between different solution methods:
            print(f"Error between end_values_rough and end_values: {np.linalg.norm(end_values_rough - end_values)}")

        # Test periodic_semilinear_pde_fdm_lirk if dim = 1
        if dim == 1:
            # Run periodic_semilinear_pde_fdm_lirk
            start_time = time.time()
            end_values_fdm = periodic_semilinear_pde_fdm_lirk(initial_values, T, laplace_factor, nonlin, space_size, nr_timesteps, dim)
            end_time = time.time()
            print(f"Time elapsed: {end_time - start_time}")
            print(f"initial_values.shape: {initial_values.shape}")
            print(f"end_values_fdm.shape: {end_values_fdm.shape}")

            # Plot results
            nr_of_trajectories = 1
            for i in range(nr_of_trajectories):
                plt.plot(np.linspace(0, space_size, nr_spacediscr), initial_values[i], label=f"Initial values {i}")
                plt.plot(np.linspace(0, space_size, nr_spacediscr), end_values[i], label=f"End values {i}")
                plt.plot(np.linspace(0, space_size, nr_spacediscr), end_values_fdm[i], label=f"End values fdm {i}")
            plt.legend()
            plt.show()

            # Calculate error between different solution methods:
            print(f"Error between end_values_fdm and end_values: {np.linalg.norm(end_values_fdm - end_values)}")



