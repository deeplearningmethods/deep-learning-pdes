import pandas as pd
import re
import openpyxl
import time
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from utils import nr_trainable_params
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from utils import flat_to_multi, numpy_to_torch, round_sig
from operator_learning_models import MyModel
from datetime import datetime
from evaluation_utils import evaluate_method_for_datatable


COLORMAP_2D = "viridis"


def savefig_safe(fig, plot_file_name):
    # I have to use this because sometimes my filenames are too long
    alternative_plot_title = plot_file_name[:-4]
    while True:
        try:
            # Save the specified figure, not just the current figure
            fig.savefig(plot_file_name, bbox_inches='tight')
            break  # Exit the loop if save is successful
        except OSError as e:
            if "[Errno 63]" in str(e):  # Check if the error is specifically a file name length issue
                if len(plot_file_name) > 10:  # Ensure the file name can be shortened
                    # Set the full file name as the plot title directly on the figure
                    fig.suptitle(alternative_plot_title, fontsize=10)
                    plot_file_name = plot_file_name[:-14] + ".pdf"  # Shorten the file name by removing the last 10 characters
                else:
                    print("Failed to save the file. File name too short to be shortened further.")
                    break
            else:
                print(f"Failed to save the file due to an unexpected error: {e}")
                break


# NOT USED ANYMORE (But could still be usefull at some point)
def csv_summary(model, csv_name):
    csv_filename = csv_name + ".csv"
    trainableParams = nr_trainable_params(model)

    first_line = ""
    data_lines = ["" for _ in range(len(model.evaluations))]

    for column_name in model.evaluations[-1]:
        first_line = first_line + column_name + ";"
        for eval_nr in range(len(model.evaluations)):
            data_lines[eval_nr] = data_lines[eval_nr] + str(model.evaluations[eval_nr][column_name]) + ";"

    with open(csv_filename, 'w+') as f:
        print(first_line + "trainable_params", file=f)
        for eval_nr in range(len(model.evaluations)):
            print(data_lines[eval_nr] + str(trainableParams), file=f)


def training_summary(model, sig=2, write_file=sys.stdout):
    trainableParams = nr_trainable_params(model)
    print(f"Total number of trainable Parameters: {trainableParams}", file=write_file)
    print(f"Total number of train steps: ", model.done_trainsteps, file=write_file)
    print(f"Optimizers: ", [optimizer for optimizer in model.optimizers], file=write_file)
    print(f"Learning rates: ", [learning_rate[:] for learning_rate in model.learning_rates], file=write_file)
    print(f"Batch sizes: ", [batch[:] for batch in model.batch_sizes], file=write_file)

    print("\n------Errors------", file=write_file)

    for e in model.evaluations:
        line = (f"{e['done_trainsteps']} tr-stps ({e['train_or_test']}):"
                f" Loss: {round_sig(e['loss_value'], sig)}"
                f", L2-err: {round_sig(e['L2_error'], sig)}"
                f", L1-err: {round_sig(e['L1_error'], sig)}"
                # f", Avg. grad: {round_sig(e['average_all_grads'], sig)}"
                f", Eval. time: {round_sig(e['eval_time'], sig)}")

        print(line, file=write_file)


def summary(model, test_input_values, test_ref_sol, plot_file_name=None, plot_title=None, plot_show=False,
            write_file=sys.stdout, sig=2):
    # Plot is not nice for 2d because it is done in 1d. But that's not really important.

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("-------------------------------------------------------------------------------------", file=write_file)
    print("--------------------------------", dt_string, "--------------------------------", file=write_file)
    print("-------------------------------------------------------------------------------------", file=write_file)

    print(f"Model: {model.modelname}\n", file=write_file)
    isadann = model.__class__.__name__ == "AdannModel"  # Somehow isinstance(model, AdannModel) does not work consistently

    if isadann:
        start_error = model.base_model.evaluations[0]["L2_error"]
        trained_base_error = model.base_model.evaluations[-1]["L2_error"]
        end_error = model.evaluations[-1]["L2_error"]

        print("Start base L2-error:", round_sig(start_error, sig), file=write_file)
        print("-------------------", file=write_file)
        print("Trained base L2-error:", round_sig(trained_base_error, sig), "  -> Factor: ",
              round_sig(start_error / trained_base_error, sig), file=write_file)
        print("Full model error:", round_sig(end_error, sig), "  -> Factor: ",
              round_sig(trained_base_error / end_error, sig), file=write_file)
        print("-------------------", file=write_file)

    else:
        start_error = model.evaluations[0]["L2_error"]
        end_error = model.evaluations[-1]["L2_error"]

        print("Start error: ", round_sig(start_error, sig), file=write_file)
        print("End error: ", round_sig(end_error, sig), "  -> Factor: ", round_sig(start_error / end_error, sig),
              file=write_file)

    print("\n-----------------TRAINING------------------", file=write_file)

    if isadann:
        print("------ TRAINING BASE------", file=write_file)
        training_summary(model.base_model, write_file=write_file)
        print("\n------ TRAINING DIFF------", file=write_file)
        training_summary(model.diff_model, write_file=write_file)

    else:
        training_summary(model, write_file=write_file)

    print("\n-----------------EXAMPLE------------------", file=write_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_input_values_torch = numpy_to_torch(test_input_values, device)
    end_predictions = model(test_input_values_torch).detach().cpu().numpy()

    fig = plt.figure()
    plt.plot(test_input_values[0], label="Input Value", color="black", linestyle="--")
    plt.plot(test_ref_sol[0], label="Reference Solution", color="black")
    plt.plot(end_predictions[0], label="Model Prediction")

    if isadann:
        base_model_prediction = model.base_model(test_input_values_torch).detach().cpu().numpy()
        scaled_difference = 1. / model.diff_factor * (test_ref_sol - base_model_prediction)
        diff_model_prediction = model.diff_model(test_input_values_torch).detach().cpu().numpy()

        # Compute diffmodel loss with numpy
        diff_model_loss = np.mean(np.square(diff_model_prediction - scaled_difference))
        print(f"Diff model loss on test set: {diff_model_loss}", file=write_file)

        plt.plot(scaled_difference[0], label="Scaled base error")
        plt.plot(diff_model_prediction[0], label="Diff model prediction")

    plt.legend()
    if plot_title is not None:
        plt.title(plot_title, fontsize=10)
    if plot_file_name is not None:
        savefig_safe(fig, plot_file_name)
        # plt.savefig(plot_file_name, bbox_inches='tight')
    if plot_show:
        plt.show()
    else:
        plt.clf()

    print("End error: ", np.sqrt(np.mean(np.square(end_predictions[0] - test_ref_sol[0]))), file=write_file)
    print("------------------------------------------------------------------------------------", file=write_file)


def plot_approx_1d(methods, methods_data, method_categories, space_grid, test_input_values_rough, test_input_values_rough_device,
                   test_ref_sol_rough, plot_folder_dir, pde_name, legend_loc=None, nr_of_plots=1):

    space_grid = space_grid[0]
    all_plots = [plt.subplots() for _ in range(nr_of_plots)]
    ol_plots = [plt.subplots() for _ in range(nr_of_plots)]
    adann_classical_plots = [plt.subplots() for _ in range(nr_of_plots)]
    method_plots = {}

    for category in method_categories:
        methods_in_category = [method for method in methods_data.index if method.startswith(category)]
        method_plots[category] = [plt.subplots() for _ in range(nr_of_plots)]

        # Check if the category are full adann models (need to check for exceptions in case the category is empty)
        try:
            is_adann = methods[methods_in_category[0]].__class__.__name__ == "AdannModel"
        except IndexError:
            is_adann = False

        colors = sns.color_palette(None, len(methods_in_category) * (3 if is_adann else 1))
        for method_nr, method in enumerate(methods_in_category):
            approx = methods[method](test_input_values_rough_device).detach().cpu().numpy()
            if is_adann:
                base_model_prediction = methods[method].base_model(test_input_values_rough_device[0:nr_of_plots]).detach().cpu().numpy()
                scaled_difference = 1. / methods[method].diff_factor * (test_ref_sol_rough[0:nr_of_plots] - base_model_prediction)
                diff_model_prediction = methods[method].diff_model(test_input_values_rough_device[0:nr_of_plots]).detach().cpu().numpy()

            for i in range(nr_of_plots):
                method_plots[category][i][1].plot(space_grid, approx[i], label=method, color=colors[method_nr * (3 if is_adann else 1)])
                if np.all(np.abs(approx[i]) < 10): # and methods_data.at[method, "L2_error"] < 0.3:
                    all_plots[i][1].plot(space_grid, approx[i], label=method)
                    if method.startswith("ADANN") or method.startswith("FDM") or method.startswith("FEM") or method.startswith("Spectral"):
                        adann_classical_plots[i][1].plot(space_grid, approx[i], label=method)
                    elif method.startswith("ANN") or method.startswith("FNO"):
                        ol_plots[i][1].plot(space_grid, approx[i], label=method)
                if is_adann:
                    method_plots[category][i][1].plot(space_grid, scaled_difference[i], label=f"{method} scaled base error", color=colors[method_nr * (3 if is_adann else 1) + 1])
                    method_plots[category][i][1].plot(space_grid, diff_model_prediction[i], label=f"{method} diff model prediction", color=colors[method_nr * (3 if is_adann else 1) + 2])

            # Plot the worst approximation
            errors = np.sqrt(np.mean(np.square(approx - test_ref_sol_rough), axis=1))
            worst_approx = np.argmax(errors)
            fig, ax = plt.subplots()
            ax.plot(space_grid, approx[worst_approx], label=method)
            ax.plot(space_grid, test_input_values_rough[worst_approx], label="Input value", color="black", linestyle="--")
            ax.plot(space_grid, test_ref_sol_rough[worst_approx], label="Reference solution", color="black")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            # Add title to figure
            fig.suptitle(f"Worst approximation for {method} - {worst_approx} \n local L2-error: {errors[worst_approx]}, overall L2-error: {methods_data.at[method, 'L2_error']}")
            savefig_safe(fig, plot_folder_dir + f"Z_worst_{method}_{pde_name}.pdf")
            # fig.savefig(plot_folder_dir + f"Z_worst_{method}_{pde_name}.pdf", bbox_inches='tight')
            # fig.show()

        for i in range(nr_of_plots):
            method_plots[category][i][1].plot(space_grid, test_input_values_rough[i], label="Input value", color="black", linestyle="--")
            method_plots[category][i][1].plot(space_grid, test_ref_sol_rough[i], label="Reference solution", color="black")

            if legend_loc is None:
                method_plots[category][i][1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            else:
                method_plots[category][i][1].legend(loc=legend_loc)
            method_plots[category][i][0].savefig(plot_folder_dir + f"{category}_plots_{pde_name}_{i}.pdf", bbox_inches='tight')
            method_plots[category][i][0].show()

    # Plot exact solution
    for i in range(nr_of_plots):
        for plot_list, name in [(all_plots, "all_plots"), (ol_plots, "ol_plots"), (adann_classical_plots, "adann_classical_plots")]:
            plot_list[i][1].plot(space_grid, test_input_values_rough[i], label="Input value", color="black", linestyle="--")
            plot_list[i][1].plot(space_grid, test_ref_sol_rough[i], label="Reference solution", color="black")
            plot_list[i][1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plot_list[i][0].savefig(plot_folder_dir + f"{name}_{pde_name}_{i}.pdf", bbox_inches='tight')
            plot_list[i][0].show()

def plot_heatmap(ax, X, Y, data, cmap, title, xlabel=r"$x_1$", ylabel=r"$x_2$"):
    if "branch" in title:
        title = title.replace("branch", "\nbranch")
    elif "kernel" in title:
        title = title.replace("kernel", "\nkernel")
    elif title.startswith("arch.:"):
        title = title.replace("arch.:", "arch.:\n")

    fontsize = 8

    heatmap = ax.pcolormesh(X, Y, data, cmap=cmap)
    ax.set_title(title, fontsize=fontsize + 2, wrap=True)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_aspect('equal')  # set X and Y axis to same scale
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(heatmap, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize)


def plot_approx_2d(methods, method_categories, space_grid, test_input_values_rough, test_input_values_rough_device,
                   test_ref_sol_rough, plot_folder_dir, pde_name, nr_of_plots=1):
    '''
        The methods in methods_approx have to appear in the same order as the method_categories
        For adanns, this looks the nicest best if there are 4 methods per category
    '''

    for j in range(nr_of_plots):
        test_input_values_rough_multi = flat_to_multi(test_input_values_rough, dim=2)[j]
        test_ref_sol_rough_multi = flat_to_multi(test_ref_sol_rough, dim=2)[j]
        X, Y = space_grid

        category_counts = {category: sum(key.startswith(category) for key in methods) for category in
                           method_categories}
        method_categories = [category for category, count in category_counts.items() if count > 0]
        isthereadann = any([methods[method].__class__.__name__ == "AdannModel" for method in methods.keys()])
        methods_per_category = max(max(category_counts.values()), 4 if isthereadann else 2)

        fig, axs = plt.subplots(nrows=1 + len(method_categories), ncols=methods_per_category,
                                figsize=(methods_per_category * 3.5, (1 + len(method_categories)) * 3.5))

        axs[0, 0].text(-0.3, 0.5, "Reference solution", transform=axs[0, 0].transAxes, rotation=90, va='center', ha='right', fontsize=14)

        plot_heatmap(axs[0, 0], X, Y, test_input_values_rough_multi, COLORMAP_2D, 'Input value')
        plot_heatmap(axs[0, 1], X, Y, test_ref_sol_rough_multi, COLORMAP_2D, 'Terminal value')

        for i in range(2, methods_per_category):
            axs[0, i].axis('off')

        for cat_nr, category in enumerate(method_categories):
            methods_in_category = sorted([method for method in methods.keys() if method.startswith(category)], key=extract_numbers)

            axs[cat_nr + 1, 0].text(-0.3, 0.5, method_categories[cat_nr], transform=axs[cat_nr + 1, 0].transAxes, rotation=90, va='center', ha='right', fontsize=14)

            if category.startswith("ADANN full"):
                # Erase unused frames
                for i in range(4, methods_per_category):
                    axs[cat_nr + 1, i].axis('off')

                for method in methods_in_category:
                    if methods[method].__class__.__name__ == "AdannModel":

                        base_model_prediction = methods[method].base_model(test_input_values_rough_device[j:j+1]).detach().cpu().numpy()
                        scaled_difference = 1. / methods[method].diff_factor * (test_ref_sol_rough[j:j+1] - base_model_prediction)
                        diff_model_prediction = methods[method].diff_model(test_input_values_rough_device[j:j+1]).detach().cpu().numpy()
                        full_model_prediction = methods[method](test_input_values_rough_device[j:j+1]).detach().cpu().numpy()

                        base_model_prediction_multi = flat_to_multi(base_model_prediction, dim=2)[0]
                        scaled_difference_multi = flat_to_multi(scaled_difference, dim=2)[0]
                        diff_model_prediction_multi = flat_to_multi(diff_model_prediction, dim=2)[0]
                        full_model_prediction_multi = flat_to_multi(full_model_prediction, dim=2)[0]

                        plot_heatmap(axs[cat_nr + 1, 0], X, Y, base_model_prediction_multi, COLORMAP_2D, f"Base model")
                        plot_heatmap(axs[cat_nr + 1, 1], X, Y, scaled_difference_multi, COLORMAP_2D, f"Scaled base error")
                        plot_heatmap(axs[cat_nr + 1, 2], X, Y, diff_model_prediction_multi, COLORMAP_2D, f"Difference model")
                        plot_heatmap(axs[cat_nr + 1, 3], X, Y, full_model_prediction_multi, COLORMAP_2D, f"Full ADANN model")

            else:
                # Erase unused frames
                for i in range(len(methods_in_category), methods_per_category):
                    axs[cat_nr + 1, i].axis('off')

                for method_nr, method in enumerate(methods_in_category):
                    approx = methods[method](test_input_values_rough_device[j:j+1]).detach().cpu().numpy()
                    approx_multi = flat_to_multi(approx, dim=2)[0]
                    subplot_title = method[len(method_categories[cat_nr]) + 2:-1]
                    plot_heatmap(axs[cat_nr + 1, method_nr], X, Y, approx_multi, COLORMAP_2D, subplot_title)

        # fig.tight_layout()
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.4)
        fig.savefig(plot_folder_dir + f"all_plots_{pde_name}_{j}.pdf", bbox_inches='tight')
        plt.show()


def create_error_vs_comptime_plot(method_categories, output_folder_dir, pde_name, fontsize = 15):
    methods_data = pd.read_csv(output_folder_dir + f'methods_data_{pde_name}.csv', sep=";", index_col="Method")

    methods_data = restore_order(methods_data, method_categories)
    methods_data.to_csv(output_folder_dir + f'methods_data_{pde_name}.csv', index=True, sep=";")

    colors = sns.color_palette(None, len(method_categories))
    category_colors = {category: color for category, color in zip(method_categories, colors)}
    markers = ['*', 'o', 's', '^', 'p', 'D', 'h', 'v', '<', '>', '+', 'x', '_']

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Average evaluation time", fontsize=fontsize)
    ax.set_ylabel(r"Estimated $L^2$-error", fontsize=fontsize)
    # ax.set_title("Estimated L2-Error vs Computational time")
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Loop over each method in the dataframe and create a scatter plot
    for category in method_categories:
        category_methods = methods_data[methods_data.index.str.startswith(category)]
        color = category_colors[category]

        for method_nr, method in enumerate(category_methods.index):
            ax.scatter(methods_data.loc[method, "test_time"], methods_data.loc[method, "L2_error"], label=method,
                       color=color, marker=markers[method_nr % len(markers)])

        if not category_methods.empty:
            ax.plot(category_methods['test_time'], category_methods['L2_error'], linestyle='-', color=color, linewidth=0.5)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(output_folder_dir + f"error_scatter_plot_{pde_name}.pdf", bbox_inches='tight')
    plt.show()


def extract_numbers(s):
    """Extract numerical values from strings, zero-pad them, and concatenate into a sortable string."""
    # Find all numbers in the string
    numbers = re.findall(r'\d+', s)
    # Zero-pad each number to 6 digits, ensuring numbers < 1 million fit
    padded_numbers = [f"{int(number):06}" for number in numbers]
    # Ensure the string has a consistent length by padding with zeros if fewer than 10 numbers are found
    padded_string = ''.join(padded_numbers).ljust(60, '0')
    return padded_string


def restore_order(methods_data, method_categories):
    # Ensure method names are strings
    methods_data.index = methods_data.index.map(str)

    # Create a temporary column for sorting by category
    methods_data['sort_category'] = pd.Categorical(
        [next((cat for cat in method_categories if method.startswith(cat)), 'Other') for method in methods_data.index],
        categories=method_categories,
        ordered=True
    )

    # Create a temporary column for sorting within categories based on numerical patterns
    methods_data['sort_numbers'] = methods_data.index.map(extract_numbers)

    # Sort by category first, then by numerical patterns within each category
    # Ensure sort_numbers is used in a way that avoids conversion to MultiIndex
    methods_data = methods_data.sort_values(by=['sort_category', 'sort_numbers'])

    # Drop the temporary columns
    methods_data = methods_data.drop(columns=['sort_category', 'sort_numbers'])

    return methods_data

# def restore_order(methods_data, method_categories):
#     # Create a temporary column for sorting
#     methods_data['sort_category'] = pd.Categorical(
#         [next((cat for cat in method_categories if method.startswith(cat)), 'Other') for method in methods_data.index],
#         categories=method_categories,
#         ordered=True
#     )
#
#     # Sort by this temporary column, and then by the index
#     methods_data = methods_data.sort_values(by=['sort_category'])
#
#     # Drop the temporary column
#     methods_data = methods_data.drop(columns=['sort_category'])
#
#     return methods_data


def evaluate_and_plot(
        methods,
        methods_data,
        method_categories,
        test_input_values_rough,
        test_ref_sol_rough,
        space_grid,
        space_size,
        output_folder_dir,
        pde_name,
        dim=1,
        nr_of_eval_runs=10,
        plot_histogram=False,
        legend_loc=None,
        nr_of_plots=1
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_input_values_rough_device = numpy_to_torch(test_input_values_rough, device)

    for name, method in methods.items():
        print("working on", name)
        evaluate_method_for_datatable(
            name,
            method,
            methods_data,
            test_input_values_rough_device,
            test_ref_sol_rough,
            space_size,
            dim,
            nr_of_eval_runs,
            plot_histogram,
            output_folder_dir
        )

    methods_data.index.name = "Method"
    sorted_methods_data = restore_order(methods_data, method_categories)
    sorted_methods_data.to_csv(output_folder_dir + f'methods_data_{pde_name}.csv', index=True, sep=";")
    print(sorted_methods_data)

    # Folder to store plots
    plots_foldername = output_folder_dir + "Sample_plots/"
    if not os.path.exists(plots_foldername):
        os.makedirs(plots_foldername)

    if dim == 1:
        plot_approx_1d(methods, methods_data, method_categories, space_grid, test_input_values_rough, test_input_values_rough_device,
                       test_ref_sol_rough, plots_foldername, pde_name, legend_loc=legend_loc, nr_of_plots=nr_of_plots)
    if dim == 2:
        plot_approx_2d(methods, method_categories, space_grid, test_input_values_rough, test_input_values_rough_device,
                       test_ref_sol_rough, plots_foldername, pde_name, nr_of_plots=nr_of_plots)

###############################################
#
# Methods to create Excel sheet
#
# Function to adjust the column width in the Excel sheet (I think I got this from chatGPT)
def auto_adjust_column_width(worksheet, dataframe, include_index=False):
    if include_index:
        # Calculate the maximum index width
        max_index_width = max(dataframe.index.astype(str).map(len).max(), len(dataframe.index.name or ''))
        worksheet.column_dimensions[openpyxl.utils.get_column_letter(1)].width = max_index_width + 2
        column_offset = 1
    else:
        column_offset = 0

    for i, col in enumerate(dataframe.columns):
        # Calculate the maximum column width
        max_width = max(dataframe[col].astype(str).map(len).max(), len(str(col)))

        # Set the column width with some padding
        worksheet.column_dimensions[openpyxl.utils.get_column_letter(i + 1 + column_offset)].width = max_width + 0.2


def save_excel_sheet(methods_data, params_dict, output_file_name):
    # Create the parameters_data DataFrame
    parameters_data = pd.DataFrame(params_dict).T
    parameters_data.index.name = "Parameter"

    # Save both DataFrames to the same sheet in an Excel file and adjust column widths
    with pd.ExcelWriter(output_file_name, engine='openpyxl') as writer:
        methods_data.to_excel(writer, index=True, sheet_name='Data', startrow=0, engine='openpyxl')
        parameters_data.to_excel(writer, index=True, sheet_name='Parameters', startrow=0, engine='openpyxl')

        # Adjust column widths
        auto_adjust_column_width(writer.sheets['Parameters'], parameters_data, include_index=True)
        auto_adjust_column_width(writer.sheets['Data'], methods_data, include_index=True)


##########################################
# Method to plot some reference solutions

def plot_reference_solutions(input_values, ref_sols, nr_reference_solutions, dim, x_values, space_size, pde_name=None,
                             output_folder_dir=None):

    input_values_multi = flat_to_multi(input_values, dim)
    ref_sol_multi = flat_to_multi(ref_sols, dim)
    nr_spacediscr = input_values_multi.shape[1]
    selected_ref_sols = range(nr_reference_solutions) if isinstance(nr_reference_solutions, int) else nr_reference_solutions

    if dim == 1:
        X = x_values(nr_spacediscr, space_size, dim=dim)
        fig = plt.figure()

        for i in selected_ref_sols:
            # Plot input value
            plt.plot(X[0], input_values_multi[i], label=f"Input Value {i + 1}")  # , color="black")
            # Plot terminal value
            plt.plot(X[0], ref_sol_multi[i], label=f"Output Value {i + 1}")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        if output_folder_dir is not None:
            fig.savefig(output_folder_dir + f"X_{pde_name}_eval_sols.pdf", bbox_inches='tight')
        plt.show()

    if dim == 2:
        X, Y = x_values(nr_spacediscr, space_size, dim=dim)
        fig = plt.figure(figsize=(20, nr_reference_solutions * 10))

        for i in selected_ref_sols:
            # Plot input value
            ax = fig.add_subplot(nr_reference_solutions, 2, 2 * i + 1)
            heatmap = ax.pcolormesh(X, Y, input_values_multi[i], cmap=COLORMAP_2D)
            ax.set_title(f'Input value {i + 1}')
            ax.set_xlabel(r"x_1")
            ax.set_ylabel(r"x_2")
            ax.set_aspect('equal')  # set X and Y axis to same scale
            plt.colorbar(heatmap, ax=ax)

            # Plot terminal value
            ax = fig.add_subplot(nr_reference_solutions, 2, 2 * (i + 1))
            heatmap = ax.pcolormesh(X, Y, ref_sol_multi[i], cmap=COLORMAP_2D)
            ax.set_title(f'Terminal value {i + 1}')
            ax.set_xlabel(r"x_1")
            ax.set_ylabel(r"x_2")
            ax.set_aspect('equal')  # set X and Y axis to same scale
            plt.colorbar(heatmap, ax=ax)

        if output_folder_dir is not None:
            fig.savefig(output_folder_dir + f"{pde_name}_eval_sols.pdf", bbox_inches='tight')
        plt.show()

    if dim == 3:
        X, Y, Z = x_values(nr_spacediscr, space_size, dim=dim)
        nr_x_slices = 8
        fig = plt.figure(figsize=(nr_x_slices * 6, nr_reference_solutions * 10))

        for i in selected_ref_sols:
            # Plot input value
            for j, xx in enumerate(range(0, nr_spacediscr, nr_spacediscr // nr_x_slices)):
                ax = fig.add_subplot(2 * nr_reference_solutions, nr_x_slices, i * 2 * nr_x_slices + j + 1)
                heatmap = ax.pcolormesh(Y[xx], Z[xx], input_values_multi[i, xx], cmap=COLORMAP_2D)
                ax.set_title(f'Input value {i + 1} for x = {X[xx][0, 0]}')
                ax.set_xlabel('y')
                ax.set_ylabel('z')
                ax.set_aspect('equal')  # set x and y axis to same scale
                plt.colorbar(heatmap, ax=ax)

            # Plot terminal value
            for j, xx in enumerate(range(0, nr_spacediscr, nr_spacediscr // nr_x_slices)):
                ax = fig.add_subplot(2 * nr_reference_solutions, nr_x_slices, (i * 2 + 1) * nr_x_slices + j + 1)
                heatmap = ax.pcolormesh(Y[xx], Z[xx], ref_sol_multi[i, xx], cmap=COLORMAP_2D)
                ax.set_title(f'Terminal value {i + 1} for x = {X[xx][0, 0]}')
                ax.set_xlabel('y')
                ax.set_ylabel('z')
                ax.set_aspect('equal')  # set x and y axis to same scale
                plt.colorbar(heatmap, ax=ax)

        if output_folder_dir is not None:
            fig.savefig(output_folder_dir + f"{pde_name}_eval_sols.pdf", bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    pass
