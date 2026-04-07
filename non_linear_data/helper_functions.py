import torch
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

def plot_data(distances,times,normalize = False):
    """
    Create a scatter plot of the data points.

    Args:
        distances: The input data points for the x-axis.
        times: The target data points for the y-axis.
        normalize: A boolean flag indicating whether the data is normalize. By default normalize = False

    """

    # Create a new figure with a specified size
    plt.figure(figsize=(8,6))

    # Plot the data points as a scatter plot
    plt.plot(distances.numpy(),times.numpy(),color = "orange",marker = "o",linestyle = "none",label = "Actual Delivery Times")

    # Check if the data is normalized to set appropriate labels and title
    if normalize:
        # Set the plot title for normalized data
        plt.title("Normalized Delivery Data (Bikes & Cars)")
        # Set the x-axis label for normalized data
        plt.xlabel("Normalized Distance")
        # Set the y-axis label for normalized data
        plt.ylabel("Normalized Time")
    # Handle the case for un-normalized data
    else:
        # Set the plot title for un-normalized data
        plt.title("Delivery Data (Bikes & Cars)")
        # Set the x-axis label for un-normalized data
        plt.xlabel("Distance (miles)")
        # Set the y-axis label for un-normalized data
        plt.ylabel("Time (minutes)")

    # Display the legend
    plt.legend()
    # Add grid to plot
    plt.grid(True)
    # Show the plot
    plt.show()


def plot_training_progress(epoch,model,distances_norm,times_norm):
    """
    Plot the training progress of model on normalized data,
    showing the current fit at each epoch

    Args:
        epoch: The current training epoch number.
        loss: The loss value at the current epoch.
        model: The model being trained.
        distances_norm: The normalized input data.
        times_norm: The normalized target data.
    """

    # Clear the previous plot from output cell
    clear_output(wait = True)

    # Make prediction using the current state of model
    predicted_norm  = model(distances_norm)

    # Convert tensors to numpy array for plotting
    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()

    # Detach predictions from the computation graph and convert to NumPy
    y_pred_plot = predicted_norm.detach().numpy()

    # Sort the data based on distance to ensure a smooth line plot
    sorted_indices = x_plot.argsort(axis = 0).flatten()

    # Create new figure for the plot
    plt.figure(figsize=(8,6))

    # Plot the original normalized data points
    plt.plot(x_plot,y_plot,color = "orange",marker = "o",linestyle = "none",label="Actual Normalized Data")

    # Plot the model's predictions as a line
    plt.plot(x_plot[sorted_indices],y_pred_plot[sorted_indices],color = "green",label = "Model Predictions")

    # Set the title of the plot, including the current epoch
    plt.title(f"Epoch: {epoch + 1} | Normalized Training Progress")

    # Set the x-axis label
    plt.xlabel("Normalized Distance")

    # Set the y-axis label
    plt.ylabel("Normalized Time")

    # Display the legend
    plt.legend()

    # Add grid to plot
    plt.grid(True)

    # Show the plot 
    plt.show()

    # Pause briefly to allow the plot to be rendered
    time.sleep(0.05)


def plot_model_fit(model,distances,times,distances_norm,times_std,times_mean):
    """
    Plots the prediction of the trained model against the original data,
    after denormalizing the predictions

    Args:
        model: The trained model used for prediction
        distance: The original, un-normalized input data.
        times: The original, un-normalized target data.
        distances_norm: The normalized input data for the model.
        times_std: The standard deviation used for de-normalization.
        times_mean: The mean value used for de-normalization.
    """

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations for prediction
    with torch.no_grad():
        predicted_norm = model(distances_norm)

    # De-normalize the predictions to their original scale
    predicted = (predicted_norm * times_std) + times_mean

    # Create a new figure for the plot
    plt.figure(figsize=(8,6))

    # Plot the original data points
    plt.plot(distances.numpy(),times.numpy(),color= "orange",marker = "o",linestyle = "none",label = "Actual Data (Bikes & Cars)")

    # Plot the de-normalized predictions from the model
    plt.plot(distances.numpy(),predicted.numpy(),color = "green",label = "Non-Linear Model Predictions")

    # Set the title of the plot
    plt.title("Non-Linear Model Fit vs. Actual Data")

    # Set the x-axis label
    plt.xlabel("Distance (miles)")

    # Set the y-axis label
    plt.ylabel("Time (minutes)")

    # Add a legend to the plot
    plt.legend()

    # Add grid to plot
    plt.grid(True)

    # Show the plot
    plt.show()



