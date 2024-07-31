import torch
import numpy as np
from utils import numpy_to_torch, torch_to_numpy
from ode_methods import SemiLinearODE, second_order_lirk

# Currently only works for dim = 1

def reaction_diffusion_pde_fdm_lirk(source_terms, T, laplace_factor, reaction_nonlin, space_size, nr_timesteps, dim=1, params=[0.5, 0.5]):
    """
        :param source_term: [batchsize, nr_spacediscr**dim] np.array
        :return: [batchsize, nr_spacediscr**dim] np.array
    """
    if dim != 1:
        raise NotImplementedError("Only dim=1 is implemented at the moment")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, nr_gridpoints = source_terms.shape
    nr_spacediscr = int(np.power(nr_gridpoints, 1/dim) + 0.5)
    assert nr_gridpoints == nr_spacediscr**dim, "Source term cannot be transformed to a square grid"
    mesh_step = space_size / nr_spacediscr
    initial_values_torch = torch.zeros((batch_size, nr_gridpoints), device=device)
    source_terms_torch = numpy_to_torch(source_terms, device) if isinstance(source_terms, np.ndarray) else source_terms

    nonlin = lambda x: reaction_nonlin(x) + source_terms_torch

    second_order_diff_quot = np.array([-2, 1] + [0 for _ in range(nr_spacediscr - 3)] + [1])
    operator = laplace_factor / mesh_step / mesh_step * np.stack(
        [np.roll(second_order_diff_quot, n) for n in range(nr_spacediscr)])
    operator_torch = numpy_to_torch(operator, device)

    semilinear_ode = SemiLinearODE(operator_torch, nonlin)

    end_values = second_order_lirk(T, semilinear_ode, initial_values_torch, nr_timesteps, params)

    return torch_to_numpy(end_values) if isinstance(source_terms, np.ndarray) else end_values


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from random_function_generators import RandnFourierSeriesGenerator

    # Test RD_pde_fdm_lirk
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)

    T = 2.
    space_size = 1.
    laplace_factor = 0.01
    reaction_rate = 0.01
    reaction_nonlin = lambda x: reaction_rate * (x - x ** 3)
    dim = 1

    batch_size = 10
    nr_spacediscr = 128
    nr_timesteps = 500

    # Create source terms
    var = 5000
    decay_rate = 2
    offset = np.power(var, 1 / decay_rate)
    inner_decay = 1.
    source_term_generator = RandnFourierSeriesGenerator([var, decay_rate, offset, inner_decay, space_size, dim])
    source_terms = source_term_generator.generate(batch_size, nr_spacediscr)

    # Run RD_pde_fdm_lirk
    start_time = time.time()
    end_values = reaction_diffusion_pde_fdm_lirk(source_terms, T, laplace_factor, reaction_nonlin, space_size, nr_timesteps, dim)
    end_time = time.time()

    print(f"Time elapsed: {end_time - start_time}")
    print(f"source_terms.shape: {source_terms.shape}")
    print(f"end_values.shape: {end_values.shape}")

    # Plot
    nr_of_trajectories = 4
    if dim == 1:
        for i in range(nr_of_trajectories):
            plt.plot(np.linspace(0, space_size, nr_spacediscr), source_terms[i], label="source_term")
            plt.plot(np.linspace(0, space_size, nr_spacediscr), end_values[i], label="end_value")
            plt.legend()
            plt.show()