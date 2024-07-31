from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import json
from utils import compare_dicts, numpy_to_torch


# Define the base class that all training samples generators should implement
class TrainingSamplesGenerator(ABC):
    def __init__(self):
        self.generator_name = None

    @abstractmethod
    def generate(self, batch_size):
        pass


# Create a generator for a Data set. For this I need my own dataset class
class DatasetFromSolutions(Dataset):
    def __init__(self, input_values, ref_solutions):
        super().__init__()
        device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_values_torch = numpy_to_torch(input_values, device) if isinstance(input_values, np.ndarray) else input_values.to(
            device)
        self.ref_solutions_torch = numpy_to_torch(ref_solutions, device) if isinstance(ref_solutions, np.ndarray) else ref_solutions.to(
            device)

    def __len__(self):
        return len(self.input_values_torch)

    def __getitem__(self, idx):
        input_value = self.input_values_torch[idx]
        ref_solution = self.ref_solutions_torch[idx]

        return input_value, ref_solution


class TrainingSamplesGeneratorFromSolutions(TrainingSamplesGenerator):
    def __init__(self, input_values, ref_solutions):
        super().__init__()
        self.generator_name = None
        self.dataset = DatasetFromSolutions(input_values, ref_solutions)
        self.Dataloaders = {}

    def generate(self, batch_size):
        # If a DataLoader with the requested batch size does not exist, create it
        if batch_size not in self.Dataloaders:
            self.Dataloaders[batch_size] = iter(DataLoader(self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True))

        # Try to get a batch from the DataLoader
        try:
            batch = next(self.Dataloaders[batch_size])
        except StopIteration:
            # If the DataLoader is exhausted, reshuffle the data and create a new DataLoader
            self.Dataloaders[batch_size] = iter(DataLoader(self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True))
            batch = next(self.Dataloaders[batch_size])

        return batch


def get_data(random_function_generator, reference_algorithm, nr_samples, nr_spacediscr, nr_timesteps, reduce_dimension,
             space_resolution_step, train_or_test, output_folder_dir=None, generate_data=True, data_load_folder=None,
             parameters=None, only_save_rough=False, save_data=False):

    if generate_data:
        if train_or_test == 'validate':
            np.random.seed(1000)
        else:
            np.random.seed(0)

        # Generate samples
        input_values_fine = random_function_generator.generate(nr_samples, nr_spacediscr)
        print("IV is generated")
        start = time.perf_counter()
        ref_sol_fine = reference_algorithm(input_values_fine, nr_timesteps)
        print("Computation time: ", time.perf_counter() - start)

        input_values_rough = reduce_dimension(input_values_fine, space_resolution_step)
        ref_sol_rough = reduce_dimension(ref_sol_fine, space_resolution_step)

        if save_data:
            # Save the data
            if only_save_rough:
                input_values_fine = input_values_rough
                ref_sol_fine = ref_sol_rough

            np.save(output_folder_dir + train_or_test + '_data_fine.npy', [input_values_fine, ref_sol_fine])
            np.save(output_folder_dir + train_or_test + '_data_rough.npy', [input_values_rough, ref_sol_rough])

    else:
        with open(data_load_folder + 'train_test_parameters.json', 'r') as fp:
            loaded_parameters = json.load(fp)

        compare_dicts(parameters, loaded_parameters)

        input_values_fine, ref_sol_fine = np.load(data_load_folder + train_or_test + '_data_fine.npy', allow_pickle=True)
        input_values_rough, ref_sol_rough = np.load(data_load_folder + train_or_test + '_data_rough.npy', allow_pickle=True)

    return input_values_fine, ref_sol_fine, input_values_rough, ref_sol_rough
