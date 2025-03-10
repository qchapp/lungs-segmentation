import matplotlib.pyplot as plt
from pathlib import Path
import json

"""
    Write a specified text to a given file.
"""
def write(content, file):
    with open(file, "a") as text_file:
        text_file.write(content)

"""
    Plot training loss and validation loss at a given epoch to a given .png file.
"""
def plot(val1, val2, epoch, LR, batch_size, epochs, plot_path):
    image_file = Path(plot_path).resolve()
    plt.plot(val1, "-b", label="training")
    plt.plot(val2, "-r", label="validation")
    plt.legend(loc="upper right")
    plt.title(f"State after epoch {epoch} with parameters:\n LR = {LR}, BATCH SIZE = {batch_size}, EPOCHS = {epochs}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(image_file)
    plt.close()

"""
    Create a space with given training info in the specified training log file.
"""
def json_creator(json_file, LR, batch_size, epochs, checkpoints_path, best_checkpoints_path, plot_path, name, bottleneck, nb_images):
    with open(json_file, "r") as file:
        data = json.load(file)

        new_data = {
            name: {
                "Success": False,
                "Learning rate": LR,
                "Batch size": batch_size,
                "Number of epochs": epochs,
                "Bottleneck channel number": bottleneck,
                "Number of images for training": nb_images,
                "Checkpoints file path": checkpoints_path,
                "Best checkpoints file path": best_checkpoints_path,
                "Plot path": plot_path,
                "Time taken": "..."
            }
        }

        data.append(new_data)
        
    with open(json_file, "w") as file:
        json.dump(data, file, indent=4)

"""
    Write the training duration for the given training.
"""
def write_time_taken(json_file, name, time_taken):
    with open(json_file, "r+") as file:
        data = json.load(file)
        data[len(data)-1][name]["Time taken"] = time_taken 
        json.dump(data, file, indent=4)
