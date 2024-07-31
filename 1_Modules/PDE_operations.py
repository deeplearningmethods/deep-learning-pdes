
import numpy as np
from utils import flat_to_multi, multi_to_flat

# Operations for discretizations for periodic semilinear heat PDEs
def x_values_periodic(nr_spacediscr, space_size = 1., dim = 1, boundary = False):
    """
    :return: [dim] list of [nr_spacediscr] np.arrays
    X1, X2, ..., Xd = x_values(nr_spacediscr, space_size, dim)
    Xq[i1, i2, ..., id] = iq * space_size / nr_spacediscr
    """
    one_d_grid = np.linspace(0, space_size, nr_spacediscr + 1) if boundary else np.linspace(0, space_size, nr_spacediscr + 1)[:-1]
    one_d_grids = [one_d_grid] * dim
    return np.meshgrid(*one_d_grids, indexing='ij')

def reduce_dimension_periodic(values, space_resolution_step, dim=1):
    values = flat_to_multi(values, dim=dim)
    slices = [slice(None)] + [slice(None, None, space_resolution_step)] * (values.ndim - 1)
    reduced_values = values[tuple(slices)]
    return multi_to_flat(reduced_values)

def get_higher_nr_spacediscr_periodic(nr_spacediscr, space_resolution_step):
    return space_resolution_step * nr_spacediscr

def create_boundary_values_periodic(function_values):
    left_values = function_values[:, 0].reshape(-1, 1)
    output_array = np.concatenate((function_values, left_values), axis=1)
    return output_array


def fdm_laplace_operator_periodic(space_size, nr_spacediscr, dim=1):
    """
    :return: [nr_spacediscr**dim, nr_spacediscr**dim] np.array
    """
    mesh_step = space_size / nr_spacediscr
    if dim == 1:
        second_order_diff_quot = np.array([-2, 1] + [0 for _ in range(nr_spacediscr - 3)] + [1])
        operator = 1. / mesh_step / mesh_step * np.stack([np.roll(second_order_diff_quot, n) for n in range(nr_spacediscr)])

    elif dim == 2:
        second_order_diff_quot = np.zeros((nr_spacediscr, nr_spacediscr))
        second_order_diff_quot[0, 0] = -4
        second_order_diff_quot[0, 1] = 1
        second_order_diff_quot[0, -1] = 1
        second_order_diff_quot[1, 0] = 1
        second_order_diff_quot[-1, 0] = 1

        operator = np.stack([np.roll(second_order_diff_quot, n, axis=1) for n in range(nr_spacediscr)])
        operator = np.stack([np.roll(operator, n, axis=1) for n in range(nr_spacediscr)])
        operator = 1. / mesh_step / mesh_step * operator
        operator = operator.reshape(nr_spacediscr**2, nr_spacediscr**2)

    else:
        raise NotImplementedError("Not implemented for dim = " + str(dim))

    return operator


def first_order_diff_matrix_trans(nr_spacediscr, space_size, diff_version=2):
    mesh_step = space_size / nr_spacediscr

    if diff_version == 0:
        first_order_diff_quot = np.array([1] + [0 for _ in range(nr_spacediscr - 2)] + [-1])
        diff_matrix = 1 / mesh_step * np.stack([np.roll(first_order_diff_quot, n) for n in range(nr_spacediscr)])
        return np.transpose(diff_matrix)

    elif diff_version == 1:
        first_order_diff_quot = np.array([-1, 1] + [0 for _ in range(nr_spacediscr - 2)])
        diff_matrix = 1 / mesh_step * np.stack([np.roll(first_order_diff_quot, n) for n in range(nr_spacediscr)])
        return np.transpose(diff_matrix)

    elif diff_version == 2:
        first_order_diff_quot = np.array([0, 1] + [0 for _ in range(nr_spacediscr - 3)] + [-1])
        diff_matrix = 1 / 2. / mesh_step * np.stack([np.roll(first_order_diff_quot, n) for n in range(nr_spacediscr)])
        return np.transpose(diff_matrix)
    else:
        raise ValueError("diff_version should be 0, 1 or 2")


if __name__=="__main__":

    print("Testing finite_difference_laplace_operator_periodic")
    print("dim = 1")
    operator = fdm_laplace_operator_periodic(3, 3, dim=1)
    print(operator)
    print(operator.shape)
    print("dim = 2")
    operator = fdm_laplace_operator_periodic(3, 3, dim=2)
    print(operator)
    print(operator.shape)

    numbers = np.arange(9).reshape(1,3,3)
    print("numbers", numbers)
    numbers_flat = multi_to_flat(numbers)
    print("numbers_flat", numbers_flat)

