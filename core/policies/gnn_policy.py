
# Import pytorch, pytorch geometric, and pandas libraries
import torch
import torch_geometric as tg

# Define a graph neural network model using pytorch geometric
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        # Define a graph convolution layer that takes 4 input features (x, y, z, layer) and outputs 16 hidden features
        self.conv1 = tg.nn.GCNConv(4, 16)
        # Define a graph convolution layer that takes 16 input features and outputs 32 hidden features
        self.conv2 = tg.nn.GCNConv(16, 32)
        # Define a linear layer that takes 32 input features and outputs 1 output feature (particle id)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, data):
        # Get the node feature matrix and the edge index matrix from the data object
        x, edge_index = data.x, data.edge_index
        # Apply the first graph convolution layer with a ReLU activation function
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        # Apply the second graph convolution layer with a ReLU activation function
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        # Apply the linear layer with a sigmoid activation function
        x = self.linear(x)
        x = torch.nn.functional.sigmoid(x)
        return x
