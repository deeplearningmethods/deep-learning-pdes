#!/bin/bash

# Execute the notebook using papermill
papermill Operator_learning_Burgers.ipynb Z_output_Operator_learning_Burgers.ipynb \
    -p test_run False

# papermill Operator_learning_Reaction_Diffusion.ipynb Z_output_Operator_learning_Reaction_Diffusion.ipynb \
#     -p test_run False

papermill Operator_learning_semilinear_heat.ipynb Z_output_Operator_learning_semilinear_heat_1d.ipynb \
    -p dim 1 \
    -p test_run False

# papermill Operator_learning_semilinear_heat.ipynb Z_output_Operator_learning_semilinear_heat_2d.ipynb \
#     -p dim 2 \
#     -p test_run False

# papermill Operator_learning_semilinear_heat.ipynb Z_output_Operator_learning_semilinear_heat_3d.ipynb \
#     -p dim 3 \
#     -p test_run False