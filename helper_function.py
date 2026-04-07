import torch
import matplotlib.pyplot as plt


def plot_results(model,distances,times):
    """
    Plots the original data points and the model's predicted line for a given dataset.
    
    Args:
        model: The trained neural network model used for making predictions.
        distances: A tensor containing the input data points (features) for which predictions are to be made.
        times: A tensor containing the actual target values (labels)
    """

    # Set the model to evaluation model to disable learning
    model.eval()

    # Disable gradient calculation for efficient inference
    with torch.no_grad():
        # Make prediction using the trained model
        predicted = model(distances)

    # Create a new figure for plot
    plt.figure(figsize = (8,6))

    # Plot the actual data points 
    plt.plot(distances.numpy(),times.numpy(),color = "orange",marker = "o",linestyle = None,label = "Actual Data")

    # Plot the predicted line from the model
    plt.plot(distances.numpy(),predicted.numpy(),color = "green",marker=None,linestyle = "-",label = "Predicted Line")

    # Set the title of the plot 
    plt.title("Actual vs Predicted Values")

    # Set the x-axis label
    plt.xlabel("Distance")

    # Set the y-axis label
    plt.ylabel("Time")

    # Display the legend to differentiate between actual and predicted data
    plt.legend()

    # Add grid to the plot 
    plt.grid(True)

    # Show the plot
    plt.show()