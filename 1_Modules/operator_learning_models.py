import torch
import torch.nn as nn
import numpy as np
import neuralop.models as fnomodels
from utils import multi_to_flat, flat_to_multi, periodic_padding_1d, periodic_padding_2d, periodic_padding_3d

# TODO: Test DeepONets. I'm not sure if they work correctly

# This is a blueprint for the models that I want to use
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.done_trainsteps = 0
        self.batch_sizes = []
        self.learning_rates = []
        self.optimizers = []
        self.modelname = None
        self.evaluations = []

    def forward(self, x):
        raise NotImplementedError("The forward method must be implemented by the subclass.")


class ANNModel(MyModel):
    """
      Model implementing ANN
    """

    def __init__(self, layer_dims):
        """
            layer_dims: is an array of natural numbers which starts and ends in nr_spacediscr
        """
        super().__init__()
        self.modelname = f"ANN (arch.: ({', '.join(str(i) for i in layer_dims)}))"
        # Init quantities
        self.layer_dims = layer_dims

        # Create ANN
        layers = []
        for i in range(len(layer_dims) - 2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, input_values):
        """
            input_values: [batch_size, space_dim]
        """
        return self.model(input_values)


class FNO1DModel(MyModel):
    """
      Model implementing FNOs for functions from R to R
    """

    def __init__(self, fno_params):
        """
            layer_dims: is an array of natural numbers which starts and ends in nr_spacediscr
        """
        super().__init__()
        # Init quantities
        self.nr_fourier_modes, self.width, self.depth = fno_params
        self.modelname = f"FNO1DModel with\n    nr_fourier_modes: {self.nr_fourier_modes}\n    width: {self.width}\n    depth: {self.depth}"

        # Create FNO
        self.fno_model = fnomodels.FNO1d(n_modes_height=self.nr_fourier_modes, hidden_channels=self.width, in_channels=1, n_layers=self.depth)

    def forward(self, input_values):
        """
            input_values: [batch_size, space_dim]
        """
        input_values = input_values.unsqueeze(1)
        pred = self.fno_model(input_values)
        return pred.squeeze(1)


class FNO2DModel(MyModel):
    """
      Model implementing FNOs for functions from R^2 to R^2
    """
    def __init__(self, fno_params):
        super().__init__()
        # Init quantities
        self.nr_fourier_modes, self.width, self.depth = fno_params
        self.modelname = f"FNO2DModel with\n    nr_fourier_modes: {self.nr_fourier_modes}\n    width: {self.width}\n    depth: {self.depth}"

        # Create FNO
        self.fno_model = fnomodels.FNO2d(n_modes_width=self.nr_fourier_modes , n_modes_height=self.nr_fourier_modes, hidden_channels=self.width, in_channels=1, n_layers=self.depth)

    def forward(self, input_values):
        """
            input_values: [batch_size, nr_spacediscr * nr_spacediscr]
        """
        input_values = flat_to_multi(input_values, dim=2)
        input_values = input_values.unsqueeze(1)
        pred = self.fno_model(input_values)
        pred.squeeze(1)
        pred = multi_to_flat(pred, dim=2)
        return pred

class FNOnDModel(MyModel):
    """
      Model implementing FNOs for functions from R^dim to R^dim
    """
    def __init__(self, nr_fourier_modes, width, depth, dim):
        super().__init__()
        # Init quantities
        self.nr_fourier_modes, self.width, self.depth, self.dim = nr_fourier_modes, width, depth, dim
        self.modelname = f"FNO (nr. modes: {nr_fourier_modes}, width: {width}, depth: {depth})"

        # Create FNO
        self.fno_model = fnomodels.FNO(n_modes=self.dim * (self.nr_fourier_modes,), hidden_channels=self.width, in_channels=1, n_layers=self.depth)

    def forward(self, input_values):
        """
            input_values: [batch_size, nr_spacediscr ** self.dim]
        """
        input_values = flat_to_multi(input_values, dim=self.dim)
        input_values = input_values.unsqueeze(1)
        pred = self.fno_model(input_values)
        pred.squeeze(1)
        pred = multi_to_flat(pred, dim=self.dim)
        return pred

    
# Implement a custom periodic convolution
class PeriodicConvnD(nn.Module):
    """n-dimensional Periodic Convolutional Layer for n = 1, 2, 3
    Returns feature maps with the same shape as the input

    in_channels : int
    out_channels : int
    kernel_size : int (Size of the kernel, should be odd. We assume that the size is the same in all dimensions)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        assert kernel_size % 2 == 1, "Please choose an odd kernel size"
        self.padding_nr = kernel_size // 2
        self.dim = dim

        if dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
            self.periodic_padding = lambda x : periodic_padding_1d(x, self.padding_nr)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            self.periodic_padding = lambda x : periodic_padding_2d(x, self.padding_nr)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
            self.periodic_padding = lambda x : periodic_padding_3d(x, self.padding_nr)
        else:
            raise ValueError("dim should be 1, 2 or 3")

    def forward(self, x):
        x = self.periodic_padding(x)
        return self.conv(x)


class CNNPeriodicnDModel(MyModel):
    """
      Model implementing n-dimensional periodic CNN model with stride one based on PeriodicConvnD layers for n = 1, 2, 3
    """

    def __init__(self, channel_dims, kernel_sizes, dim):
        """
            channel_dims : is an array of natural numbers which starts and ends in 1
            kernel_sizes : list of kernel sizes
            dim: 1, 2, 3

        """
        assert dim in [1, 2, 3], "dim should be 1, 2 or 3"
        assert channel_dims[0] == channel_dims[-1] == 1, "channel_dims should start and end with 1"

        super().__init__()
        self.modelname = f"Periodic CNN (arch.: ({', '.join(str(i) for i in channel_dims)}), kernel sizes: ({', '.join(str(i) for i in kernel_sizes)}))"

        # Init quantities
        self.channel_dims = channel_dims
        self.kernel_sizes = kernel_sizes
        self.dim = dim

        # Create CNN
        conv_layers = []
        for i in range(len(channel_dims) - 2):
            conv_layers.append(PeriodicConvnD(channel_dims[i], channel_dims[i+1], kernel_sizes[i], dim))
            conv_layers.append(nn.GELU())
        conv_layers.append(PeriodicConvnD(channel_dims[-2], channel_dims[-1], kernel_sizes[-1], dim))
        self.model = nn.Sequential(*conv_layers)

    def forward(self, input_values):
        """
            input_values: [batch_size, nr_spacediscr, ..., nr_spacediscr]
        """
        input_values = flat_to_multi(input_values, dim=self.dim)
        input_values = input_values.unsqueeze(1)
        pred = self.model(input_values)
        pred = pred.squeeze(1)
        return multi_to_flat(pred, dim=self.dim)


# Implement a custom full stride convolution
class FullStrideConvnD(nn.Module):
    """Full stride Convolutional Layer
    Returns feature maps with the dimensions of input / kernel_size

    kernel_size : int (should divide dimension of input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        if dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, stride=kernel_size)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, stride=kernel_size)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=0, stride=kernel_size)
        else:
            raise ValueError("dim should be 1, 2 or 3")

    def forward(self, x):
        return self.conv(x)

# Implement a custom deconvolution/ transpose convolution
class FullStrideDeconvnD(nn.Module):
    """Full stride Deconvolutional Layer
    Returns feature maps with the dimensions of input * kernel_size

    kernel_size : int
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        if dim == 1:
            self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=kernel_size)
        elif dim == 2:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=kernel_size)
        elif dim == 3:
            self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=kernel_size)
        else:
            raise ValueError("dim should be 1, 2 or 3")

    def forward(self, x):
        return self.deconv(x)


class CNNEncDec(MyModel):
    """
      Model implementing periodic CNN model with full stride convolutions and deconvolutions
    """

    def __init__(self, channel_dims, kernel_sizes, dim):
        """
            channel_dims : is an array of (typically increasing) natural numbers which starts in 1
            kernel_sizes : list of kernel sizes. Product of all kernel sizes should devide the input dimension of the model
            dim : 1, 2, 3

        """
        if dim > 3:
            raise ValueError("dim should be 1, 2 or 3")

        super().__init__()
        self.modelname = f"Enc.-Dec. CNN (arch.: ({', '.join(str(i) for i in channel_dims)}), kernel sizes: ({', '.join(str(i) for i in kernel_sizes)}))"

        # Init quantities
        self.channel_dims = channel_dims
        self.kernel_sizes = kernel_sizes
        self.dim = dim

        # Create ANN
        layers = []
        # Encoder Layers
        for i in range(len(channel_dims)-1):
            layers.append(FullStrideConvnD(channel_dims[i], channel_dims[i + 1], kernel_sizes[i], dim))
            layers.append(nn.GELU())
        # Decoder Layers
        for i in range(1, len(channel_dims)-1):
            layers.append(FullStrideDeconvnD(channel_dims[-i], channel_dims[-i-1], kernel_sizes[-i], dim))
            layers.append(nn.GELU())

        layers.append(FullStrideDeconvnD(channel_dims[1], channel_dims[0], kernel_sizes[0], dim))
        self.model = nn.Sequential(*layers)

    def forward(self, input_values):
        """
            input_values: [batch_size, nr_spacediscr^dim]
        """
        input_values = flat_to_multi(input_values, dim=self.dim)
        input_values = input_values.unsqueeze(1)
        pred = self.model(input_values)
        pred = pred.squeeze(1)
        return multi_to_flat(pred, dim=self.dim)


class DeepONet(MyModel):
    """
      Model implementing DeepONets
    """

    def __init__(self, trunk_architecture, branch_architecture, eval_points=None):
        """
            trunk_architecture[0] = nr_spacediscr**space_dim
            trunk_architecture[-1] = branch_architecture[-1]
            branch_architecture[0] = space_dim
            eval_points : [nr_eval_points, space_dim] Torch Tensor
        """
        super().__init__()
        self.modelname = f"DeepONet (trunk arch.: ({', '.join(str(i) for i in trunk_architecture)}), branch arch.: ({', '.join(str(i) for i in branch_architecture)}))"

        # Init quantities
        self.trunk_architecture = trunk_architecture
        self.branch_architecture = branch_architecture
        self.eval_points = eval_points

        # Create Trunk
        trunk_layers = []
        for i in range(len(trunk_architecture) - 2):
            trunk_layers.append(nn.Linear(trunk_architecture[i], trunk_architecture[i+1]))
            trunk_layers.append(nn.GELU())
        trunk_layers.append(nn.Linear(trunk_architecture[-2], trunk_architecture[-1]))
        self.trunk_model = nn.Sequential(*trunk_layers)

        # Create Branch
        branch_layers = []
        for i in range(len(branch_architecture) - 2):
            branch_layers.append(nn.Linear(branch_architecture[i], branch_architecture[i + 1]))
            branch_layers.append(nn.GELU())
        branch_layers.append(nn.Linear(branch_architecture[-2], branch_architecture[-1]))
        self.branch_model = nn.Sequential(*branch_layers)

        # Bring eval points into right form
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.eval_points is not None:
            self.eval_points = torch.Tensor(eval_points).to(device).type(torch.float32)


    def forward(self, input_values, eval_points=None):
        """
            input_values : [batch_size, nr_spacediscr**space_dim]
            eval_points : [nr_eval_points, space_dim] Torch Tensor
        """

        if eval_points is None:
            eval_points = self.eval_points

        trunk = self.trunk_model(input_values)
        branch = self.branch_model(eval_points)

        return torch.einsum('ik,jk->ij', trunk, branch)


if __name__ == "__main__":

    fno2d_params = [10, 30, 4]
    fno2d = FNO2DModel(fno2d_params)
    input_model = multi_to_flat(torch.rand((5, 32, 32)))
    output_model = fno2d(input_model)
    print(input_model.shape, output_model.shape)

    in_channels = 2
    out_channels = 2
    kernel_size = 5
    batch_size = 5
    nr_spacediscr = 32
    for dim in [1, 2, 3]:
        print(f"Testing PeriodicConvnD for dim {dim}")
        periodic_conv = PeriodicConvnD(in_channels, out_channels, kernel_size, dim)
        input_layer = torch.rand((batch_size, in_channels, nr_spacediscr, nr_spacediscr, nr_spacediscr)[:dim+2])
        output_layer = periodic_conv(input_layer)
        print(f"    input_layer.shape: {input_layer.shape}\n    output_layer.shape: {output_layer.shape}")

    channel_dims = [1, 16, 32, 16, 1]
    kernel_sizes = [3, 5, 5, 3]
    batch_size = 5
    nr_spacediscr = 32
    for dim in [1, 2, 3]:
        print(f"Testing CNNPeriodicnDModel for dim {dim}")
        cnn_periodic_model = CNNPeriodicnDModel(channel_dims, kernel_sizes, dim)
        input_model = torch.rand((batch_size, nr_spacediscr, nr_spacediscr, nr_spacediscr)[:dim+1])
        input_model_flat = multi_to_flat(input_model, dim=dim)
        output_model_flat = cnn_periodic_model(input_model_flat)
        output_model = flat_to_multi(output_model_flat, dim=dim)
        print(f"    input_model.shape: {input_model.shape}\n    output_model.shape: {output_model.shape}")

    in_channels = 2
    out_channels = 2
    kernel_size = 4
    batch_size = 5
    nr_spacediscr = 32
    for dim in [1, 2, 3]:
        print(f"Testing FullStrideConvnD for dim {dim}")
        fullstride_conv = FullStrideConvnD(in_channels, out_channels, kernel_size, dim)
        fullstride_deconv = FullStrideDeconvnD(out_channels, in_channels, kernel_size, dim)
        input_layer = torch.rand((batch_size, in_channels, nr_spacediscr, nr_spacediscr, nr_spacediscr)[:dim+2])
        middle_layer = fullstride_conv(input_layer)
        end_layer = fullstride_deconv(middle_layer)

        print(f"    input_layer.shape: {input_layer.shape}")
        print(f"    middle_layer.shape: {middle_layer.shape}  (should be {(batch_size, out_channels, nr_spacediscr//kernel_size, nr_spacediscr//kernel_size, nr_spacediscr//kernel_size)[:dim+2]})")
        print(f"    end_layer.shape: {end_layer.shape}  (should be {(batch_size, in_channels, nr_spacediscr, nr_spacediscr, nr_spacediscr)[:dim+2]})")

    channel_dims = [1, 16, 32, 64]
    kernel_sizes = [2, 4, 4]
    batch_size = 5
    nr_spacediscr = 32
    for dim in [1, 2, 3]:
        print(f"Testing CNNEncDec for dim {dim}")
        cnn_periodic_model = CNNEncDec(channel_dims, kernel_sizes, dim)
        input_model = torch.rand((batch_size, nr_spacediscr, nr_spacediscr, nr_spacediscr)[:dim+1])
        input_model_flat = multi_to_flat(input_model, dim=dim)
        output_model_flat = cnn_periodic_model(input_model_flat)
        output_model = flat_to_multi(output_model_flat, dim=dim)
        print(f"    input_model.shape: {input_model.shape}\n    output_model.shape: {output_model.shape}")


    nr_spacediscr = 32
    batch_size = 5
    nr_eval_points = 100
    for dim in [1, 2, 3]:
        print(f"Testing DeepONet for dim {dim}")
        trunk_architecture = [nr_spacediscr**dim, 20, 40, 50]
        branch_architecture = [dim, 20, 40, 50]
        eval_points = np.random.randn(nr_eval_points, dim)
        deeponet = DeepONet(trunk_architecture, branch_architecture, space_dim=dim, eval_points=eval_points)
        input_model = torch.rand((batch_size, nr_spacediscr**dim))
        output_model = deeponet(input_model)
        print(f"    input_model.shape: {input_model.shape}")
        print(f"    output_model.shape: {output_model.shape} (should be {(batch_size, nr_eval_points)})")
