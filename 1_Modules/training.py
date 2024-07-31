import torch
import numpy as np
from timer import Timer
import time
import os
import sys
import matplotlib.pyplot as plt
import copy
import seaborn as sns


from documentation_utils import summary
from evaluation_utils import evaluate
from utils import print_to_file_and_console

colormap = "RdYlBu"
def track_optimization_params(model, learning_rate=None, batch_size=None, optimizer=None, output=True):
    # Keep track of batch sizes
    if batch_size is not None:
        if len(model.batch_sizes) == 0:
            model.batch_sizes = model.batch_sizes + [[model.done_trainsteps, batch_size]]
            if output:
                print("    Initial batch size: ", batch_size)
        elif model.batch_sizes[-1][1] != batch_size:
            model.batch_sizes = model.batch_sizes + [[model.done_trainsteps, batch_size]]
            if output:
                print("    New batch size: ", batch_size)

    if learning_rate is not None:
        if len(model.learning_rates) == 0:
            model.learning_rates = model.learning_rates + [[model.done_trainsteps, learning_rate]]
            if output:
                print("    Initial learning rate: ", learning_rate)
        elif model.learning_rates[-1][1] != learning_rate:
            model.learning_rates = model.learning_rates + [[model.done_trainsteps, learning_rate]]
            if output:
                print("    New learning rate: ", learning_rate)

    if optimizer is not None:
        if len(model.optimizers) == 0:
            model.optimizers = model.optimizers + [[model.done_trainsteps, optimizer]]
        elif model.optimizers[-1][1] != optimizer:
            model.optimizers = model.optimizers + [[model.done_trainsteps, optimizer]]
            if output:
                print("    New optimizer: ", optimizer)


def update_training_parameters_old(loss_value, current_lr, current_bs, best_loss, number_of_steps_without_improvement,
                               nr_changes_without_improvement, optimizer, model, tolerance, max_batchsize, output=True):
    if loss_value < best_loss:
        best_loss = loss_value
        number_of_steps_without_improvement = 0
        nr_changes_without_improvement = 0
    elif number_of_steps_without_improvement < tolerance:
        number_of_steps_without_improvement += 1
    elif number_of_steps_without_improvement == tolerance:
        if nr_changes_without_improvement == 1:
            print("Training ended due to insufficient improvement")
            return None, None, None, None, None
        else:
            current_lr = current_lr / 5.
            current_bs = min([current_bs * 2, max_batchsize])
            nr_changes_without_improvement = nr_changes_without_improvement + 1
            number_of_steps_without_improvement = 0

            track_optimization_params(model, current_lr, current_bs, output=output)

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

    return current_lr, current_bs, best_loss, number_of_steps_without_improvement, nr_changes_without_improvement


def update_training_parameters(model, optimizer, last_eval_loss, new_eval_loss, current_lr, current_bs, max_batchsize, eval_steps, relative_improvement_tolerance=0.96, output=True):
    # Check relative improvement of model
    if new_eval_loss > relative_improvement_tolerance * last_eval_loss:
        # If we just updated parameters after last evaluation, we should abort training
        if model.learning_rates[-1][0] == model.done_trainsteps - eval_steps:
            print(f"Training ended at trainstep {model.done_trainsteps} due to insufficient improvement")
            return None, None
        else:
            current_lr = current_lr / 5.
            current_bs = min([current_bs * 2, max_batchsize])
            track_optimization_params(model, current_lr, current_bs, output=output)

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

    return current_lr, current_bs

def whole_training(
        model,
        training_samples_generator,
        optimizer_class,
        loss_fn,
        initial_lr,
        initial_batchsize,
        eval_steps=10,
        improvement_tolerance=0.96,
        max_trainsteps=1000,
        max_batchsize=2048,
        output_steps=100,
        output=True,
        lr_search_params=(100, -20, 5., 30),
        validation_input_values=None,
        validation_ref_sol=None,
        instance_identifier="",
        output_file=sys.stdout,
        output_folder=""
):
    training_timer = Timer()
    training_timer.start()

    if max_trainsteps == 0:
        return 0, None, initial_lr

    if validation_input_values is None:
        validation_input_values, validation_ref_sol = training_samples_generator.generate(initial_batchsize)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set up optimizer
    if initial_lr is None:
        initial_lr = find_optimal_learningrate_with_documentation(
            model,
            training_samples_generator,
            optimizer_class,
            loss_fn,
            initial_batchsize,
            validation_input_values,
            validation_ref_sol,
            lr_search_params,
            identifier=instance_identifier,
            output_file=output_file,
            output_folder=output_folder,
        )
    optimizer = optimizer_class(model.parameters(), lr=initial_lr)

    # Variables for the training loop
    current_lr = initial_lr
    current_bs = initial_batchsize
    last_eval_loss = float('inf')
    new_eval_loss = None

    track_optimization_params(model, current_lr, current_bs, optimizer_class, output)

    # The training loop
    model.train()
    loss = None
    for t in range(max_trainsteps):

        # Evaluate model and update learning rate and batch size
        if t % eval_steps == 0:
            # Evaluate model
            new_eval_loss = evaluate(model, validation_input_values, validation_ref_sol, space_size=1., train_or_test="validate")**2
            if np.isnan(new_eval_loss) or np.isinf(new_eval_loss):
                training_timer.stop()
                if output:
                    print(f"Training ended at trainstep {model.done_trainsteps} due to {'nan' if np.isnan(new_eval_loss) else 'inf'} validation loss")
                return training_timer.read_time(), None, initial_lr

            if output:
                print(f"{t} (validation loss) : {new_eval_loss:.9g}")

            current_lr, current_bs = update_training_parameters(model, optimizer, last_eval_loss, new_eval_loss, current_lr, current_bs, max_batchsize, eval_steps, improvement_tolerance, output=output)
            last_eval_loss = new_eval_loss

        if current_lr is None:
            break

        # Generate training samples
        input_values, reference_solutions = training_samples_generator.generate(current_bs)
        input_values, reference_solutions = input_values.to(device), reference_solutions.to(device)

        # Compute prediction error
        pred = model(input_values)
        loss = loss_fn(pred, reference_solutions)

        # Stop if loss is nan
        if torch.isnan(loss):
            training_timer.stop()
            print(f"Training ended at trainstep {model.done_trainsteps} due to nan loss")
            return training_timer.read_time(), loss if loss is None else loss.item(), initial_lr

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model.done_trainsteps % output_steps == 0 and output:
            print(f"{model.done_trainsteps} (train loss) : {loss.detach().cpu().item():.9g}")

        model.done_trainsteps += 1

    training_timer.stop()
    return training_timer.read_time(), loss if loss is None else loss.item(), initial_lr

def golden_section_search(f, a, b, rel_tol=None, max_iter=10):
    # Golden section search for finding the minimum of an unimodal function
    # It is assumed that one nan value always implies that all higher values also return nan
    # f: a function to minimize
    # a, b: the interval to search
    # tol: the tolerance for the minimum
    # max_iter: the maximum number of iterations

    # Golden ratio
    rho = (3 - np.sqrt(5)) / 2

    # Loop to create initial values until at least f_c is not nan
    f_c = np.nan
    f_d = np.nan
    c = b
    while np.isnan(f_d) and np.isnan(f_c):
        b = c

        # Initialize
        c = a + rho * (b - a)
        d = b - rho * (b - a)

        # Evaluate
        print(f"\rEvaluating at {c}", end="")
        f_c = f(c)
        print(f"\rEvaluating at {d}", end="")
        f_d = f(d)

    function_evaluations = []
    evaluation_points = []
    function_evaluations += [f_c, f_d]
    evaluation_points += [c, d]

    # Loop
    for i in range(max_iter):
        if f_c < f_d or np.isnan(f_d):
            b = d
            d = c
            c = a + rho * (b - a)
            f_d = f_c
            print(f"\rEvaluating at {c}", end="")
            f_c = f(c)
            function_evaluations.append(f_c)
            evaluation_points.append(c)
        else:
            a = c
            c = d
            d = b - rho * (b - a)
            f_c = f_d
            print(f"\rEvaluating at {d}", end="")
            f_d = f(d)
            function_evaluations.append(f_d)
            evaluation_points.append(d)

        if (rel_tol is not None) and ((b - a) / 2) / ((b + a) / 2) < rel_tol:
            break

    return (a + b) / 2, function_evaluations, evaluation_points


def find_optimal_learning_rate_golden_section_search(
        model,
        training_samples_generator,
        optimizer_class,
        loss_fn,
        batch_size,
        validate_input_values,
        validate_ref_sol,
        search_parameters=(100, -20, 5., 30),
        rel_tol=None,
        training_output=False,
        output_steps=10
):
    '''
    Find the optimal learning rate for a given model using golden section search
    This should be done before training the model. Doing it afterward might lose the training
    '''

    nr_trainsteps, smallest_power, largest_power, maxiter = search_parameters

    start_error = evaluate(model, validate_input_values, validate_ref_sol, train_or_test="validate")**2

    # Check if model can be restored and if so, save it in a variable
    restore_possible = False
    try:
        model.restore_initialization()
        restore_possible = True
    except:
        pass

    # Do golden section search
    def error_for_learning_rate(learning_rate_power):
        learning_rate = 10**learning_rate_power
        if restore_possible:
            model.restore_initialization()
            test_model = model
        else:
            test_model = copy.deepcopy(model)

        whole_training(
            model=test_model,
            training_samples_generator=training_samples_generator,
            optimizer_class=optimizer_class,
            loss_fn=loss_fn,
            initial_lr=learning_rate,
            initial_batchsize=batch_size,
            eval_steps=nr_trainsteps+10,
            max_trainsteps=nr_trainsteps,
            max_batchsize=batch_size,
            output_steps=output_steps,
            output=training_output
        )

        error = evaluate(test_model, validate_input_values, validate_ref_sol, train_or_test="validate")**2
        return error

    best_lr_power, errors, learning_rate_powers = golden_section_search(error_for_learning_rate, smallest_power, largest_power, rel_tol, maxiter)
    best_lr = 10**best_lr_power
    learning_rates = 10**np.array(learning_rate_powers)

    if restore_possible:
        model.restore_initialization()

    return best_lr, start_error, errors, learning_rates


def find_optimal_learningrate_with_documentation(
        model,
        training_samples_generator,
        optimizer_class,
        loss_fn,
        batchsize,
        validation_input_values,
        validation_ref_sol,
        lr_search_parameters=(100, -30, 5., 30),
        identifier=None,
        output_file=sys.stdout,
        output_folder="",
        plot_title=None
):

    print_to_file_and_console(f"------ SEARCHING OPTIMAL LEARNING RATE FOR {identifier} ------", file=output_file)
    if plot_title == None:
        plot_title = identifier

    best_learning_rate, start_error, errors, learning_rates = find_optimal_learning_rate_golden_section_search(
        model,
        training_samples_generator,
        optimizer_class,
        loss_fn,
        batchsize,
        validation_input_values,
        validation_ref_sol,
        lr_search_parameters,
        rel_tol=None,
        training_output=False,
        output_steps=1
    )

    print_to_file_and_console(f"\nBest learning rate for {identifier}: {best_learning_rate}", file=output_file)
    print("Errors:\n", errors, file=output_file)
    print("Learning rates:\n", learning_rates, file=output_file)

    # Plot the errors
    plt.figure()
    colors = sns.color_palette(None, 3)
    plt.axhline(y=start_error, color=colors[0], linestyle='-', label='Loss at initialization', zorder=0)
    plt.axvline(x=best_learning_rate, color=colors[1], linestyle='-', label='Approximate best learning rate', zorder=0)
    plt.scatter(learning_rates, errors, color=colors[2], label='Errors', zorder=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(plot_title)
    plt.xlabel('Learning rate')
    plt.ylabel(f"Loss after {lr_search_parameters[0]} train steps")
    plt.legend()
    plt.savefig(output_folder + f"/Z_learning_rates_search_{identifier}.pdf", bbox_inches='tight')
    plt.close()

    return best_learning_rate


def create_and_train_models(
        modelclass,
        list_params,
        training_samples_generator,
        optimizer_class,
        loss_fn,
        training_kwargs,
        lr_search_params,
        nr_runs,
        methods,
        methods_data,
        foldername,
        test_input_values,
        test_ref_sol,
        validation_input_values=None,
        validation_ref_sol=None,
        pde_name="",
        local_learning_rates=False,
        dim=1
):
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    if validation_input_values is None:
        validation_input_values = test_input_values
        validation_ref_sol = test_ref_sol

    with open(foldername + f'/Simulation_results_{pde_name}.txt', 'w+') as f:

        for params in list_params:
            start_time = time.perf_counter()

            # Prepare placeholder
            errors = float('inf') * np.ones(nr_runs)
            errors_validation = float('inf') * np.ones(nr_runs)
            train_times = np.zeros_like(errors)
            best_model = None
            initial_learning_rates = np.zeros_like(errors)

            print_to_file_and_console("\n\n", file=f)
            if training_kwargs['initial_lr'] is None and not local_learning_rates:
                model = modelclass(*params)
                name = model.modelname
                best_lr = find_optimal_learningrate_with_documentation(
                    model,
                    training_samples_generator,
                    optimizer_class,
                    loss_fn,
                    training_kwargs["initial_batchsize"],
                    validation_input_values,
                    validation_ref_sol,
                    lr_search_params,
                    identifier=f"{name}_overall_lr",
                    output_file=f,
                    output_folder=foldername,
                )
                initial_lr = best_lr/2.
                training_kwargs = training_kwargs.copy()
                training_kwargs["initial_lr"] = initial_lr
                print_to_file_and_console(f"Choosing nonlocal learning rate: {initial_lr}", file=f)

            for run_nr in range(nr_runs):
                model = modelclass(*params)
                name = model.modelname

                print_to_file_and_console(f"\n--------------------------------------RUN {run_nr} for {name} --------------------------------------", file=f)

                # Train the model
                train_times[run_nr], last_loss, initial_learning_rates[run_nr] = whole_training(
                    model=model,
                    training_samples_generator=training_samples_generator,
                    optimizer_class=optimizer_class,
                    loss_fn=loss_fn,
                    **training_kwargs,
                    output=True,
                    lr_search_params=lr_search_params,
                    validation_input_values=validation_input_values,
                    validation_ref_sol=validation_ref_sol,
                    instance_identifier=f"{name}_nr_{run_nr}",
                    output_file=f,
                    output_folder=foldername
                )

                print("\n -------------------------------------------")
                print("    Last loss: ", last_loss)
                print("    Train time: ", train_times[run_nr])
                print("\n -------------------------------------------")

                # Evaluate the model
                errors_validation[run_nr] = evaluate(model, validation_input_values, validation_ref_sol, space_size=1, dim=dim, loss_fn=loss_fn, train_or_test="validate")
                errors[run_nr] = evaluate(model, test_input_values, test_ref_sol, space_size=1, dim=dim, loss_fn=loss_fn, train_or_test="test")

                print("\nSummary of model", file=f)
                summary(model, test_input_values[0:1], test_ref_sol[0:1], plot_file_name=f"{foldername}/Z_plot_{name}_{run_nr}.pdf", write_file=f)
                print("\n\n", file=f)

                # Save model if it is the best
                if errors_validation[run_nr] == np.min(errors_validation):
                    best_model = model
                    print(" We have a new best model!")



            # Save best model
            methods[best_model.modelname] = best_model
            methods_data.at[best_model.modelname, "training_time"] = total_time = time.perf_counter() - start_time

            # Print some summary values
            print_to_file_and_console("--------------------------------------", file=f)
            print_to_file_and_console(f"Best test model (nr. {np.argmin(errors)}): {np.min(errors)}", file=f)
            print_to_file_and_console(f"Best validation model (nr. {np.argmin(errors_validation)}): {np.min(errors_validation)}", file=f)
            print_to_file_and_console("\nTrain times:", file=f)
            print_to_file_and_console(f"    Total time : {total_time}", file=f)
            print_to_file_and_console(f"    Total train time: {np.sum(train_times)}", file=f)
            print_to_file_and_console(f"    Average train time: {np.mean(train_times)}", file=f)
            print_to_file_and_console("--------------------------------------", file=f)
            print_to_file_and_console("\n\n", file=f)

            print(f"---------------------------------------------------------------------------------------\n\n\n")
            print(f"Summary of best {best_model.modelname}: run_nr {np.argmin(errors)}:\n")
            summary(best_model, test_input_values[0:1], test_ref_sol[0:1], plot_file_name=foldername + f"/Best_trained_{best_model.modelname}_{pde_name}.pdf", plot_show=False)

            # Save error data
            np.savetxt(foldername + f"/Y_end_errors_{best_model.modelname}.txt", errors)
            np.savetxt(foldername + f"/Y_end_errors_validation_{best_model.modelname}.txt", errors_validation)
            np.savetxt(foldername + f"/Y_initial_learning_rates_{best_model.modelname}.txt", initial_learning_rates)


if __name__ == "__main__":
    # Test golden_section_search
    def f(x):
        print("Evaluated at: ", x)
        return x**2 + x

    min, fe, ep = golden_section_search(f, -1, 1, 1e-4, 30)
    print("Minimum: ", min)

    print(fe)
    print(ep)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.scatter(ep, fe)
    plt.show()
