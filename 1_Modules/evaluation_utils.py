import time
import torch
import torch.nn as nn
import numpy as np
from utils import nr_trainable_params
import matplotlib.pyplot as plt
from utils import flat_to_multi, numpy_to_torch, torch_to_numpy, round_sig
from operator_learning_models import MyModel


def evaluate(model:MyModel, eval_input_values, eval_ref_sol, space_size=1, dim=1, loss_fn=nn.MSELoss(), train_or_test="test"):
    # Currently this only works for periodic functions and dim=1

    evaluation = {
        "done_trainsteps": model.done_trainsteps,
        "train_or_test": train_or_test
    }

    # print(f"I just evaluated {model.__class__.__name__} with {train_or_test}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    eval_input_values_device = numpy_to_torch(eval_input_values, device) if isinstance(eval_input_values, np.ndarray) else eval_input_values.to(device)
    eval_ref_sol_device = numpy_to_torch(eval_ref_sol, device) if isinstance(eval_ref_sol, np.ndarray) else eval_ref_sol.to(device)

    with torch.no_grad():
        start_time = time.perf_counter()
        _ = model(eval_input_values_device)
        evaluation["eval_time"] = time.perf_counter() - start_time

    predictions = model(eval_input_values_device)
    loss_value = loss_fn(predictions, eval_ref_sol_device)

    evaluation["loss_value"] = loss_value.item()
    evaluation["L2_error"] = np.sqrt(space_size ** dim * evaluation["loss_value"]) #This would have to be changed for non-periodic functions

    loss_by_data = torch.mean(torch.square(predictions - eval_ref_sol_device), 1)
    norm_error_by_data = torch.sqrt(space_size ** dim * loss_by_data)
    evaluation["L1_error"] = torch.mean(norm_error_by_data).item()

    # grads = torch.autograd.grad(loss_value, model.parameters())
    # average_value_of_grads = np.zeros(len(grads))

    # for g in range(len(grads)):
    #     average_value_of_grads[g] = torch.mean(torch.abs(grads[g])).item()

    # evaluation["average_all_grads"] = np.mean(average_value_of_grads)

    model.evaluations = model.evaluations + [evaluation]

    return evaluation["L2_error"]


def evaluate_method_for_datatable(name, method, methods_data, test_input_values_rough_device, test_ref_sol_rough, space_size, dim, nr_of_eval_runs = 10, plot_histogram = False, ouput_folder_dir=None):

    warm_up = 2
    with torch.no_grad():
        eval_times = np.zeros(nr_of_eval_runs + warm_up) # Add two runs because often the first run is very different
        for run_nr in range(nr_of_eval_runs + warm_up):
            start_time_eval = time.perf_counter()
            method(test_input_values_rough_device)
            eval_times[run_nr] = time.perf_counter() - start_time_eval

        methods_data.at[name, "test_time"] = np.mean(eval_times[warm_up:])
        if plot_histogram:
            assert ouput_folder_dir is not None, "Need to provide output_folder_dir to plot histogram"
            plt.figure()
            plt.hist(eval_times[warm_up:])
            plt.title(f"Histogram for eval times for {name}")
            plt.savefig(ouput_folder_dir + f"Z_hist_eval_times_{name}.pdf", bbox_inches='tight')

        approx = method(test_input_values_rough_device)
        approx = approx.detach().cpu().numpy()
        methods_data.at[name, "L2_error"] = np.sqrt(space_size ** dim * np.mean(np.square(test_ref_sol_rough - approx)))

    if isinstance(method, torch.nn.Module) and not name.startswith(("FDM", "FEM", "Spectral")):
        methods_data.at[name, "nr_params"] = nr_trainable_params(method)
        methods_data.at[name, "done_trainsteps"] = method.done_trainsteps
        methods_data.at[name, "learning_rate_history"] = method.learning_rates
        methods_data.at[name, "batch_size_history"] = method.batch_sizes
    else:
        methods_data.at[name, "nr_params"] = 0
        methods_data.at[name, "done_trainsteps"] = 0
        methods_data.at[name, "learning_rate_history"] = 0
        methods_data.at[name, "batch_size_history"] = 0

    return approx


##########################################
def eval_method_memory_safe(method, data, divisions=1):
    """
    Attempts to evaluate the method with the provided data.
    If there's an OOM error, it splits the data into two halves,
    evaluates each half, and then concatenates the results.

    Args:
    - method (torch.nn.Module): the model or method to evaluate.
    - data (torch.Tensor): the data to use for evaluation.

    Returns:
    - approx (np.array): the model's output.
    - duration (float): time taken to produce the output.
    """

    try:
        data_chunks = torch.chunk(data, 2 ** (divisions - 1), dim=0)
        print(f"Trying with {data_chunks[0].shape}")
        approx = method(data_chunks[0])
        for data_chunk in data_chunks[1:]:
            approx = torch.cat([approx, method(data_chunk)], axis=0)
        return approx
    except RuntimeError as e:
        if 'out of memory' in str(e):
            # Clearing CUDA cache
            torch.cuda.empty_cache()
            print(f"Ran out of memory. Try again with {divisions + 1}")

            return eval_method_memory_safe(method, data, divisions + 1)
        else:
            raise e


if __name__ == "__main__":
    pass

