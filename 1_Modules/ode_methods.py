import numpy as np
import torch
from scipy.linalg import expm
import time


#######################################################################################
#######################################################################################
'''                                ODE CLASSES                                      '''
#######################################################################################
#######################################################################################

class ODE:
    def __init__(self, ode_name = None, space_dim = None, ode_dynamic = None):
        self.ode_name = ode_name
        self.space_dim = space_dim
        self.ode_dynamic = ode_dynamic

    def dynamic(self, inputs):
        return self.ode_dynamic(inputs)


class SemiLinearODE(ODE):

    def __init__(self, linear_operator, nonlin, nonlin_name = "no name"):
        super().__init__()
        self.Linear_operator = linear_operator
        self.nonlin = nonlin
        self.nonlin_name = nonlin_name

        self.space_dim, _ = linear_operator.shape
        self.ode_name =  f"{self.space_dim}-dimensional semilinear ODE with \n    A = {self.Linear_operator} \n    nonlin: {self.nonlin_name}"

    def dynamic(self, inputs):
        return np.matmul(inputs, np.transpose(self.Linear_operator)) + self.nonlin(inputs)


class SemiLinearODEDiagOperator(ODE):

    def __init__(self, linear_operator_diag, nonlin, nonlin_name = "no name"):
        super().__init__()
        self.linear_operator_diag = linear_operator_diag
        self.nonlin = nonlin
        self.nonlin_name = nonlin_name
        self.space_dim = linear_operator_diag.shape[0]

        self.ode_name = f"{self.space_dim}-dimensional semilinear ODE with diag operator \n    A = {self.linear_operator_diag} \n    nonlin: {self.nonlin_name}"

    def dynamic(self, inputs):
        return inputs * self.linear_operator_diag + self.nonlin(inputs)


class SemiLinearFEMODE(ODE):
    def __init__(self, mass_operator, stiffness_operator, nonlin, nonlin_name="no name"):
        super().__init__()
        self.mass_operator = mass_operator
        self.stiffness_operator = stiffness_operator
        self.nonlin = nonlin
        self.nonlin_name = nonlin_name
        self.space_dim, _ = stiffness_operator.shape

        self.ode_name = f"{self.space_dim}-dimensional semilinear ODE with \n    Mass_operator = {self.mass_operator}\n    Stiffness_operator: {self.stiffness_operator}\n    nonlin: {self.nonlin_name}"

    def dynamic(self, inputs):
        raise NotImplementedError("This method is not implemented yet")
        return 0


#######################################################################################
#######################################################################################
'''                                  METHODS                                        '''
#######################################################################################
#######################################################################################


def explicit_euler(T, ode: ODE, initial_values, nr_timesteps):
    timestep_size = float(T)/nr_timesteps
    u = initial_values
    for m in range(nr_timesteps):
        u = u + timestep_size * ode.dynamic(u)
    return u


def second_order_lirk(T, semilinear_ode: SemiLinearODE, initial_values, nr_timesteps, params):
    '''
        initial_values : [batch_size, ode.space_dim]
        Usage of p1 and p2 are according to the latex document
    '''
    # Check which library to use
    np_or_torch = np if isinstance(initial_values, np.ndarray) else torch

    p1 = params[0]
    p2 = params[1]
    timestep_size = float(T)/nr_timesteps

    if np_or_torch == np:
        implicit_flow_trans = np.transpose(np.linalg.inv(np.eye(semilinear_ode.space_dim) - timestep_size * p2 * semilinear_ode.Linear_operator))
        operator_trans = np.transpose(semilinear_ode.Linear_operator)
    else:
        identity_matrix = torch.eye(semilinear_ode.space_dim, device=semilinear_ode.Linear_operator.device)
        implicit_flow_trans = torch.inverse(identity_matrix - timestep_size * p2 * semilinear_ode.Linear_operator).t()
        operator_trans = semilinear_ode.Linear_operator.t()

    u = initial_values
    for m in range(nr_timesteps):
        k_one = np_or_torch.matmul(np_or_torch.matmul(u, operator_trans) + semilinear_ode.nonlin(u), implicit_flow_trans)
        k_two = np_or_torch.matmul(np_or_torch.matmul(u + 2 * timestep_size * p1 * (0.5 - p2) * k_one, operator_trans) + semilinear_ode.nonlin(u + timestep_size * p1 * k_one), implicit_flow_trans)

        u = u + timestep_size * ((1 - 1./(2 * p1)) * k_one + 1./(2 * p1) * k_two)
    return u


def second_order_lirk_diag_operator(T, semilinear_ode: SemiLinearODEDiagOperator, initial_values, nr_timesteps, params):
    '''
        initial_values : [batch_size, ode.space_dim]
        Usage of p1 and p2 are according to the latex document
    '''
    np_or_torch = np if isinstance(initial_values, np.ndarray) else torch

    p1 = params[0]
    p2 = params[1]
    timestep_size = float(T)/nr_timesteps

    implicit_flow_diag = np_or_torch.reciprocal(1 - timestep_size * p2 * semilinear_ode.linear_operator_diag)
    operator_diag = semilinear_ode.linear_operator_diag

    u = initial_values
    for m in range(nr_timesteps):
        k_one = (u * operator_diag + semilinear_ode.nonlin(u)) * implicit_flow_diag
        k_two = ((u + 2 * timestep_size * p1 * (0.5 - p2) * k_one) * operator_diag + semilinear_ode.nonlin(u + timestep_size * p1 * k_one)) * implicit_flow_diag

        u = u + timestep_size * ((1 - 1./(2 * p1)) * k_one + 1./(2 * p1) * k_two)
    return u


def second_order_lirk_fem(T, semilinear_fem_ode : SemiLinearFEMODE, initial_values, nr_timesteps, params):
    '''
        initial_values : [batch_size, ode.space_dim]
        Usage of p1 and p2 are according to the latex document
    '''
    np_or_torch = np if isinstance(initial_values, np.ndarray) else torch

    p1 = params[0]
    p2 = params[1]
    timestep_size = float(T)/nr_timesteps

    if np_or_torch == np:
        stiffness_trans = np.transpose(semilinear_fem_ode.stiffness_operator)
        implicit_flow_trans = np.transpose(np.linalg.inv(semilinear_fem_ode.mass_operator - timestep_size * p2 * semilinear_fem_ode.stiffness_operator))
    else:
        stiffness_trans = semilinear_fem_ode.stiffness_operator.t()
        implicit_flow_trans = torch.linalg.inv(semilinear_fem_ode.mass_operator - timestep_size * p2 * semilinear_fem_ode.stiffness_operator).t()

    u = initial_values
    for m in range(nr_timesteps):
        b_one = np_or_torch.matmul(u, stiffness_trans) + semilinear_fem_ode.nonlin(u)
        k_one = np_or_torch.matmul(b_one, implicit_flow_trans)
        b_two = np_or_torch.matmul(u + 2 * p1 * (0.5 - p2) * timestep_size * k_one, stiffness_trans) + semilinear_fem_ode.nonlin(u + timestep_size * p1 * k_one)
        k_two = np_or_torch.matmul(b_two, implicit_flow_trans)
        u = u + timestep_size * ((1 - 1./(2 * p1)) * k_one + 1./(2 * p1) * k_two)
    return u


##############################
# Splitting methods

def strang_splitting(first_flow, second_flow, initial_values, nr_timesteps):
    '''
        initial_values : [batch_size, space_dim]
    '''
    u = initial_values
    for m in range(nr_timesteps):
        u = first_flow(u)
        u = second_flow(u)
        u = first_flow(u)
    return u

def strang_splitting_semilinear(T, semilinear_ode: SemiLinearODE, initial_values, nr_timesteps, params, implicit_first=True):
    '''
        initial_values : [batch_size, ode.space_dim]
        Use exponential for linear part. Later might have to change this to rational approximations such as crank-nicolson
    '''
    np_or_torch = np if isinstance(initial_values, np.ndarray) else torch

    p1 = params[0]

    timestep = float(T)/nr_timesteps
    implicit_timestep = timestep/2. if implicit_first else timestep
    explicit_timestep = timestep if implicit_first else timestep/2.

    # Compute implicit flow for linear part with matrix exponential
    if np_or_torch == np:
        implicit_flow_trans = np.transpose(expm((implicit_timestep * semilinear_ode.Linear_operator)))
    else:
        implicit_flow_trans = torch.matrix_exp(implicit_timestep * semilinear_ode.Linear_operator).t()

    def implicit_step(u):
        return np_or_torch.matmul(u,implicit_flow_trans)

    def explicit_step(u):
        k_one = semilinear_ode.nonlin(u)
        k_two = semilinear_ode.nonlin(u + explicit_timestep * p1 * k_one)
        return u + explicit_timestep * ((1 - 1./(2 * p1)) * k_one + 1./(2 * p1) * k_two)

    if implicit_first:
        return strang_splitting(implicit_step, explicit_step, initial_values, nr_timesteps)
    else:
        return strang_splitting(explicit_step, implicit_step, initial_values, nr_timesteps)

def strang_splitting_semilinear_2(T, semilinear_ode: SemiLinearODE, initial_values, nr_timesteps, params):
    '''
        initial_values : [batch_size, ode.space_dim]
        Use exponential for linear part. Later I might have to change this to rational approximations such as crank-nicolson
    '''
    np_or_torch = np if isinstance(initial_values, np.ndarray) else torch

    p1 = params[0]

    timestep = float(T)/nr_timesteps

    # Compute implicit flow for linear part with matrix exponential
    if np_or_torch == np:
        implicit_flow_trans = np.transpose(expm((timestep/2. * semilinear_ode.Linear_operator)))
    else:
        implicit_flow_trans = (timestep * semilinear_ode.Linear_operator).expm().t()

    def implicit_step(u):
        return np_or_torch.matmul(u,implicit_flow_trans)

    def explicit_step(u):
        k_one = semilinear_ode.nonlin(u)
        k_two = semilinear_ode.nonlin(u + timestep * p1 * k_one)
        return u + timestep * ((1 - 1./(2 * p1)) * k_one + 1./(2 * p1) * k_two)

    u = initial_values
    for m in range(nr_timesteps):
        u = implicit_step(u)
        u = explicit_step(u)
        u = implicit_step(u)
    return u


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test the explicit euler
    print("Test explicit_euler")
    ode = ODE(ode_name="Test ODE", space_dim = 2, ode_dynamic = lambda x: x)
    initial_values = np.array([1, 2])
    T = 1
    nr_timesteps = 100
    print(explicit_euler(T, ode, initial_values, nr_timesteps))

    ################################################
    fig_errors, ax_errors = plt.subplots(1, 1)
    ax_errors.set_xscale('log')
    ax_errors.set_yscale('log')
    ax_errors.set_xlabel('Number of timesteps')
    ax_errors.set_ylabel('Error')


    # Test methods for semilinear ODEs
    dim = 10
    semilinear_ode = SemiLinearODE(np.random.rand(dim, dim), lambda x: np.sin(x))
    initial_values = np.random.rand(dim)
    T = 1
    nr_timesteps = [2**i for i in range(2, 10)]
    log_nr_timesteps = np.log(np.array(nr_timesteps))

    # Reference soutions with second_order_lirk
    params = [0.5, 0.5]
    ref_nr_timesteps = 2**12
    ref_solution = second_order_lirk(T, semilinear_ode, initial_values, ref_nr_timesteps, params)

    # Test second_order_lirk
    print("Test second_order_lirk")
    params = [0.5, 0.5]
    # Compute all errors, scatter plot them
    errors = [np.linalg.norm(second_order_lirk(T, semilinear_ode, initial_values, nr_timestep, params) - ref_solution) for nr_timestep in nr_timesteps]
    ax_errors.scatter(nr_timesteps, errors, label="second_order_lirk")
    # Estimate rate
    log_errors = np.log(np.array(errors))
    slope = (log_errors[-1] - log_errors[0]) / (log_nr_timesteps[-1] - log_nr_timesteps[0])
    print("Estimated rate in time: ", slope)

    # Test the strang splitting
    print("Test strang_splitting_semilinear")
    params = [0.5]
    for implicit_first in [True, False]:
        errors = [np.linalg.norm(strang_splitting_semilinear(T, semilinear_ode, initial_values, nr_timestep, params, implicit_first) - ref_solution) for nr_timestep in nr_timesteps]
        ax_errors.scatter(nr_timesteps, errors, label=f"strang_splitting_semilinear implicit_first={implicit_first}")
        log_errors = np.log(np.array(errors))
        slope = (log_errors[-1] - log_errors[0]) / (log_nr_timesteps[-1] - log_nr_timesteps[0])
        print(f"Estimated rate in time for implicit_first={implicit_first}: ", slope)

    fig_errors.legend()
    fig_errors.show()

    # Test the second order lirk with diag operator
    print("Test second_order_lirk_diag_operator")
    semilinear_ode_diag = SemiLinearODEDiagOperator(np.array([1, 1]), lambda x: x)
    initial_values = np.array([1, 2])
    T = 1
    nr_timesteps = 100
    params = [0.5, 0.5]
    print(second_order_lirk_diag_operator(T, semilinear_ode_diag, initial_values, nr_timesteps, params))


    # Test the second order lirk for
    print("Test second_order_lirk_fem")
    mass_operator = np.array([[1, 0], [0, 1]])
    stiffness_operator = np.array([[1, 0], [0, 1]])
    semilinear_fem_ode = SemiLinearFEMODE(mass_operator, stiffness_operator, lambda x: x)
    initial_values = np.array([1, 2])
    T = 1
    nr_timesteps = 100
    params = [0.5, 0.5]
    print(second_order_lirk_fem(T, semilinear_fem_ode, initial_values, nr_timesteps, params))