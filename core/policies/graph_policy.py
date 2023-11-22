# Import pytorch, pytorch geometric, and pandas libraries
import torch
import torch_geometric as tg
import pandas as pd
from pathlib import Path
import numpy as np
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from policies.gnn_policy import GNN
from utiles import getFileList


# Define a function that calculates the dphi between two hits
def dphi(hit1, hit2):
    # Get the x and y coordinates of the hits
    x1, y1 = hit1[2], hit1[3]
    x2, y2 = hit2[2], hit2[3]
    # Calculate the phi angles of the hits
    phi1 = np.arctan2(y1, x1)
    phi2 = np.arctan2(y2, x2)
    # Calculate the difference between the phi angles
    dphi = phi2 - phi1
    # Wrap the difference to the range [-pi, pi]
    dphi = np.remainder(dphi + np.pi, 2 * np.pi) - np.pi
    return dphi


indir ="/home/amohamed/ML/LUXETrackML/output_luxe_data/"
if __name__=="__main__":
    # Define the paths to the files that contain the hits and particles information
    hitfiles = getFileList(indir,1,5,"hits_%04d.csv")
    particlefiles = getFileList(indir,1,5,"particles_%04d.csv")

    hits_df = pd.concat(map(pd.read_csv, hitfiles), ignore_index=True)
    particles_df = pd.concat(map(pd.read_csv, particlefiles),ignore_index=True)

    # Load the files as pandas dataframes
    # hits_df = pd.read_csv(hits_file)
    # particles_df = pd.read_csv(particles_file)

    # Convert the dataframes to pytorch tensors
    hits_tensor = torch.from_numpy(hits_df.values)
    particles_tensor = torch.from_numpy(particles_df.values)

    # Create a list of graphs, one for each event
    graphs = []

    # Iterate over the events
    for event_id in hits_df["event_id"].unique():
    # Get the hits and particles for the current event
        hits = hits_tensor[hits_tensor[:, 0] == event_id]
        particles = particles_tensor[particles_tensor[:, 0] == event_id]

        # Get the number of hits and particles for the current event
        n_hits = hits.shape[0]
        n_particles = particles.shape[0]

        # Create a node feature matrix for the current event, consisting of the hit positions and layer numbers
        x = hits[:, 2:6]


        # Create an edge index matrix for the current event, consisting of the pairs of hit ids that belong to the same particle
        edge_index = torch.empty((2, n_particles * 10), dtype=torch.long)
        edge_index[0] = particles[:, 1].repeat_interleave(10)
        edge_index[1] = particles[:, 2:12].flatten()

        # Create an edge attribute matrix for the current event, consisting of the particle ids that connect the hits
        edge_attr = particles[:, 1].repeat_interleave(10).unsqueeze(1)

        # Create a node label vector for the current event, consisting of the particle ids that each hit belongs to
        y = particles[:, 1].repeat_interleave(10)

        # Create a graph object for the current event using pytorch geometric
        graph = tg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    #   Append the graph object to the list of graphs
        graphs.append(graph)

    # Create a dataset object from the list of graphs using pytorch geometric
    dataset = tg.data.Batch.from_data_list(graphs)
    # Create an instance of the graph neural network model
    model = GNN()

    # Define a loss function using pytorch
    loss_fn = torch.nn.BCELoss()

    # Define an optimizer using pytorch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Define a number of training epochs
    epochs = 100

    # Train the model using pytorch
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        # Initialize the loss value to zero
        loss = 0
        # Iterate over the batches in the dataset
        for batch in dataset:
            # Get the node feature matrix, the edge index matrix, and the node label vector from the batch object
            x, edge_index, y = batch.x, batch.edge_index, batch.y
            # Zero the gradients of the optimizer
            optimizer.zero_grad()
            # Forward pass the data through the model and get the output
            output = model(batch)
            # Compute the loss value using the output and the label
            loss += loss_fn(output, y)
            # Backward pass the loss value and update the gradients
            loss.backward()
            # Update the model parameters using the optimizer
            optimizer.step()
        # Print the epoch number and the loss value
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")



# Convert the dataframes to pytorch tensors
hits_tensor = torch.from_numpy(hits_df.values)
particles_tensor = torch.from_numpy(particles_df.values)

# Create a list of graphs, one for each event
graphs = []

# Iterate over the events
for event_id in hits_df["event_id"].unique():
        
    # Get the hits and particles for the current event
    hits = hits_tensor[hits_tensor[:, 0] == event_id]
    particles = particles_tensor[particles_tensor[:, 0] == event_id]

    # Get the number of hits and particles for the current event
    n_hits = hits.shape[0]
    n_particles = particles.shape[0]

    # Create a node feature matrix for the current event, consisting of the hit positions and layer numbers
    x = hits[:, 2:6]

    # Create an empty list to store the edge index matrix
    edge_index = []

    # Iterate over the particles for the current event
    for particle in particles:
    # Get the particle id and the hit ids from the particle tensor
        particle_id = particle[1]
        hit_ids = particle[2:12]
        # Iterate over the hit ids
        for hit_id in hit_ids:
            # Get the hit from the hits tensor
            hit = hits[hits[:, 1] == hit_id][0]
            # Check if the hit is valid (not zero)
            if hit_id != 0:
                # Iterate over the other hit ids
                for other_hit_id in hit_ids:
                    # Get the other hit from the hits tensor
                    other_hit = hits[hits[:, 1] == other_hit_id][0]
                    # Check if the other hit is valid and different from the current hit
                    if other_hit_id != 0 and other_hit_id != hit_id:
                        # Calculate the dphi between the two hits using a custom function
                        dphi_value = dphi(hit, other_hit)
                        # Check if the dphi is less than 0.01 rad using a custom criterion
                        if abs(dphi_value) < 0.01:
                            # Add an edge between the two hits with their particle id as an attribute
                            edge_index.append([hit_id, other_hit_id, particle_id])
                            # Convert the edge index list to a pytorch tensor
                            edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    # Create a node label vector for the current event, consisting of the particle ids that each hit belongs to
    y = particles[:, 1].repeat_interleave(10)

    # Create a graph object for the current event using pytorch geometric
    graph = tg.data.Data(x=x, edge_index=edge_index, y=y)

    # Append the graph object to the list of graphs
    graphs.append(graph)

# Create a dataset object from the list of graphs using pytorch geometric
dataset = tg.data.Batch.from_data_list(graphs)
