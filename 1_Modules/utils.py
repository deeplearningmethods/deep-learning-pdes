import numpy as np
import pandas as pd
import torch
from datetime import datetime
import os
import json

def round_sig(f, p):
    return float(f"{f:.{p}e}")

def create_output_folder(pde_name):
    # get current date and time
    now = datetime.now()

    # format as a string: year-month-day-hours-minutes
    output_folder = now.strftime(f"Z Outputs/ZZ %Y-%m-%d %Hh%Mm%Ss {pde_name}")

    # create the new directory
    os.makedirs(output_folder, exist_ok=True)
    output_folder_dir = output_folder + "/"

    return output_folder_dir


# Functions to read and save lists from/to files
def save_list_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

def read_list_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# Function to
def print_to_file_and_console(text, file):
    print(text)
    print(text, file=file)



# Function to compare to dictionaries. Used when loading data
def compare_dicts(param1, param2):
    keys1 = set(param1.keys())
    keys2 = set(param2.keys())
    common_keys = keys1 & keys2

    diff_keys = keys1 ^ keys2  # keys present in only one of the paramionaries
    diff_values = {k: (param1.get(k), param2.get(k)) for k in common_keys if param1.get(k) != param2.get(k)}

    if diff_keys:
        print("Keys present in only one of the dictionaries:", diff_keys)
    if diff_values:
        print("The parameters differ in values:")
        for key, (val1, val2) in diff_values.items():
            print(f"For key '{key}', got {val1} in param1 and {val2} in param2")
    if not diff_keys and not diff_values:
        print("The parameters are the same.")

#Swap order of keys in dictionary
def swap_order(dictionary, key1, key2):
    # Check if both keys are in the dictionary
    if key1 not in dictionary or key2 not in dictionary:
        raise ValueError("Both keys must be in the dictionary")

    new_dict = {}
    for key in dictionary:
        new_key = key1 if key == key2 else key2 if key == key1 else key
        new_dict[new_key] = dictionary[new_key]

    return new_dict


#Swap order of rows in pd dataframe
def swap_rows(df, index1, index2):
    # Check if both indices are in the DataFrame
    if index1 not in df.index or index2 not in df.index:
        raise ValueError("Both indices must be in the DataFrame")

    rows = []
    for idx in df.index:
        new_idx = index1 if idx == index2 else index2 if idx == index1 else idx
        rows.append(df.loc[new_idx:new_idx])

    # Concatenate all rows to form the new DataFrame
    new_df = pd.concat(rows)

    return new_df


#########################################
# Padding tools for perdiodic convolutions

def periodic_padding_1d(x, padding_nr):
    return torch.cat((x[:, :, -padding_nr:], x, x[:, :, 0:padding_nr]), dim=2)

def periodic_padding_2d(x,padding_nr):
    x = torch.cat((x[:, :, -padding_nr:, :], x, x[:, :, 0:padding_nr, :]), dim=2)
    x = torch.cat((x[:, :, :, -padding_nr:], x, x[:, :, :, 0:padding_nr]), dim=3)
    return x

def periodic_padding_3d(x,padding_nr):
    x = torch.cat((x[:, :, -padding_nr:, :, :], x, x[:, :, 0:padding_nr, :, :]), dim=2)
    x = torch.cat((x[:, :, :, -padding_nr:, :], x, x[:, :, :, 0:padding_nr, :]), dim=3)
    x = torch.cat((x[:, :, :, :, -padding_nr:], x, x[:, :, :, :, 0:padding_nr]), dim=4)
    return x



#########################################
# Multidimensional tools
def fftfreq_multi(modes_per_dim, dim, d=1.0):
    """
    Generate a grid of multi-dimensional frequencies for an FFT

    Parameters:
    shape (tuple): The shape of the input array for which to compute the frequencies.
    d (float): The sample spacing for the input array. (This should actually depend on the dimension too, but

    Returns:
    numpy.ndarray: An array of arrays representing the FFT frequencies in each dimension with shape (dim, *shape)
    fftfreq_multi(*modes_per_dim, d=1.0)[q, i1, i2, ...] = (iq modulo modes_per_dim[q]) / (d * modes_per_dim[q])
    where the modulus is taken to be cut off at the center of the interval. See https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
    """

    freqs = [np.fft.fftfreq(modes_per_dim, d) for _ in range(dim)]
    grids = np.array(np.meshgrid(*freqs, indexing='ij'))
    return grids

def flat_to_multi(values, dim=2):

    batch_size = values.shape[0]
    grid_size_per_dim = int(round(values.shape[1] ** (1.0 / dim)))

    # Check if conversion is possible
    assert grid_size_per_dim ** dim == values.shape[1], f"Conversion not possible with values.shape {values.shape}, dim {dim}, grid_size_per_dim {grid_size_per_dim}"

    # New shape: (batch_size, grid_size_per_dim, ..., grid_size_per_dim)
    new_shape = [batch_size] + [grid_size_per_dim] * dim
    return values.reshape(new_shape)

def multi_to_flat(values, dim=2):
    # Flatten all dimensions except the batch dimension
    return values.reshape(values.shape[0], -1)


def pad_zeros_in_middle(modes, end_size):
    # Assuming modes shape is [batch_size, n, ..., n]
    assert end_size >= modes.size(1), "end_size must be greater than or equal to the size of modes"
    assert modes.size(1) % 2 == 0, "nr_modes must be divisible by 2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Calculate the number of zeros to be added in the middle
    middle_padding = end_size - modes.size(1)

    # Padding process for each dimension
    for d in range(1, modes.dim()):
        # Split the tensor into two parts
        split_index = modes.size(d) // 2
        first_part = modes.narrow(d, 0, split_index)
        second_part = modes.narrow(d, split_index, modes.size(d) - split_index)

        # Create a zero tensor for padding in the middle
        zero_pad_shape = list(modes.shape)
        zero_pad_shape[d] = middle_padding
        zero_pad = torch.zeros(zero_pad_shape, dtype=modes.dtype, device=device)

        # Concatenate the parts with the zero padding in the middle
        modes = torch.cat([first_part, zero_pad, second_part], dim=d)

    return modes

def remove_middle(modes, end_size):
    # Assuming modes shape is [batch_size, n, ..., n]
    assert end_size <= modes.size(1), "nr_spacediscr must be greater than or equal to the size of modes"
    assert modes.size(1) % 2 == 0, "nr_modes must be divisible by 2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    half_length = end_size // 2
    for d in range(1, modes.dim()):
        dim_size = modes.shape[d]

        # Concatenate slices from beginning and end parts of the tensor
        modes = torch.cat(
            (modes.index_select(d, torch.arange(0, half_length, device=device)),
             modes.index_select(d, torch.arange(dim_size-half_length, dim_size, device=device))),
            dim=d)

    return modes


# Torch tools
def torch_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def numpy_to_torch(array, device="cpu"):
    return torch.from_numpy(array).float().to(device)

def nr_trainable_params(model):
    return np.sum([p.numel() for p in model.parameters() if p.requires_grad])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from random_function_generators import RandnFourierSeriesGeneratorStartControl

    # Test fftfreq_multi
    freqs = fftfreq_multi(10, 3)
    i1, i2, i3 = np.random.randint(0, 9), np.random.randint(0, 9), np.random.randint(0, 9)
    print(freqs[0, i1, i2, i3], freqs[1, i1, i2, i3], freqs[2, i1, i2, i3])
    print("Should be", i1 / 10, i2 / 10, i3 / 10)

    #############################
    # Test pad_zeros_in_middle
    batch_size = 1
    nr_modes = 2
    nr_spacediscr = 4
    dim = 3

    # Create a sample tensor
    modes = torch.randn(batch_size, * [nr_modes] * dim)

    # Apply padding
    padded_modes = pad_zeros_in_middle(modes, nr_spacediscr)
    print("modes:", modes)
    print("padded_modes:", padded_modes)
    print("padded_modes.shape:", padded_modes.shape)
    print(f"Expected shape: {[batch_size] + dim * [nr_spacediscr]}")

    #############################
    # Test remove_middle
    new_modes = remove_middle(padded_modes, nr_modes)
    print("new_modes:", new_modes)
    print("new_modes.shape:", new_modes.shape)
    print(f"Expected shape: {[batch_size] + dim * [nr_modes]}")

    #############################
    # Test pad_zeros_in_middle and remove_middle in the context of fft

    # Initial values setup
    space_size = 2 * np.pi
    dim = 1
    var = 500
    decay_rate = 2
    offset = np.power(var, 1 / decay_rate)
    inner_decay = 1.
    start_var = 0.2
    initial_value_generator = RandnFourierSeriesGeneratorStartControl([var, decay_rate, offset, inner_decay, space_size, start_var, dim])


    batch_size = 1
    nr_spacediscr = 64
    nr_modes = 32

    x_values_normal = np.linspace(0, space_size, nr_spacediscr, endpoint=False)
    x_values_rough = np.linspace(0, space_size, nr_modes, endpoint=False)

    # Test removing middle
    initial_values = torch.tensor(initial_value_generator.generate(batch_size, nr_spacediscr), dtype=torch.float32)
    modes = torch.fft.fftn(initial_values, dim=list(range(1, dim+1)), norm="forward")

    # Remove middle
    reduced_modes = remove_middle(modes, nr_modes)
    reconstructed_initial_values = torch.fft.ifftn(reduced_modes, dim=list(range(1, dim+1)), norm="forward").real

    # Plot original initial values and reconstructed initial values
    plt.plot(x_values_normal, initial_values[0, :].numpy(), label="original")
    plt.plot(x_values_rough, reconstructed_initial_values[0, :].numpy(), label="reconstructed")
    plt.legend()
    plt.show()

    # Test padding
    initial_values = torch.tensor(initial_value_generator.generate(batch_size, nr_modes), dtype=torch.float32)
    modes = torch.fft.fftn(initial_values, dim=list(range(1, dim+1)), norm="forward")

    # Pad zeros in the middle
    padded_modes = pad_zeros_in_middle(modes, nr_spacediscr)
    reconstructed_initial_values = torch.fft.ifftn(padded_modes, dim=list(range(1, dim+1)), norm="forward").real

    # Plot original initial values and reconstructed initial values
    plt.plot(x_values_rough, initial_values[0, :].numpy(), label="original")
    plt.plot(x_values_normal, reconstructed_initial_values[0, :].numpy(), label="reconstructed")
    plt.legend()
    plt.show()
