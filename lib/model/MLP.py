import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_GAN(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_channels,
                 merge_layer=0,
                 res_layers=[],
                 norm='group'):
        super(MLP_GAN, self).__init__()

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.merge_layer = merge_layer if merge_layer > 0 else len(hidden_channels) // 2
        self.res_layers = res_layers
        self.norm = norm

        # Generator layers
        self.filters.append(nn.Conv1d(input_dim, hidden_channels[0], 1))
        for l in range(0, len(hidden_channels) - 1):
            if l in self.res_layers:
                self.filters.append(nn.Conv1d(
                    hidden_channels[l] + input_dim,
                    hidden_channels[l + 1],
                    1))
            else:
                self.filters.append(nn.Conv1d(
                    hidden_channels[l],
                    hidden_channels[l + 1],
                    1))
            if l != len(hidden_channels) - 2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, hidden_channels[l + 1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(hidden_channels[l + 1]))

        # Output layer
        self.output_layer = nn.Conv1d(hidden_channels[-1], output_dim, 1)

    def forward(self, noise):
        '''
        Generate 3D shapes from input noise (generator forward pass).
        args:
            noise: [B, input_dim, N]
        return:
            [B, output_dim, N] generated 3D shapes
        '''
        y = noise
        tmpy = noise
        for i, f in enumerate(self.filters):
            y = f(
                y if i not in self.res_layers
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters) - 1:
                if self.norm not in ['batch', 'group']:
                    y = F.leaky_relu(y)
                else:
                    y = F.leaky_relu(self.norms[i](y))
            if i == self.merge_layer:
                phi = y.clone()

        y = self.output_layer(y)

        return y, phi

# Initialize the MLP_GAN with appropriate dimensions and hyperparameters
input_dim = 100  # Adjust this based on the size of the input noise vector
output_dim = 3  # Adjust this based on the dimension of the generated 3D shapes
hidden_channels = [64, 128, 256, 512]  # Adjust this based on the desired hidden layer sizes
merge_layer = 2  # Adjust this if you want the generator to provide intermediate features

# Create the generator
generator = MLP_GAN(input_dim, output_dim, hidden_channels, merge_layer=merge_layer)

# Create the discriminator (use the same architecture as the generator)
discriminator = MLP_GAN(output_dim, 1, hidden_channels, merge_layer=merge_layer)

# Note that the output_dim of the discriminator is 1 to represent the probability of the shape being real or fake.
