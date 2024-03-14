import os
import json
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def process_image(file_path):
    """Read JSON file and return its data as a numpy array."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return np.array(data)

def create_heatmap(input_folder):
    """Create a heatmap from multiple JSON files representing images using parallel processing."""
    # Function to process each file
    def process_file(filename):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            return process_image(file_path)
        return None

    # Initialize an array to store the summed values
    sum_array = None

    # Create a thread pool executor
    with ThreadPoolExecutor() as executor:
        # Create futures for each file in the input folder
        futures = [executor.submit(process_file, filename) for filename in os.listdir(input_folder)]
        
        # Wait for the futures to complete and aggregate the results
        for future in futures:
            image_array = future.result()
            if image_array is not None:
                if sum_array is None:
                    # Initialize sum_array with the first image's dimensions
                    sum_array = np.zeros_like(image_array)
                sum_array += image_array

    # Check if sum_array was initialized (in case there were no valid JSON files)
    if sum_array is not None:
        # normalize the sum_array
        sum_array = sum_array/sum_array.max()
        # Save the heatmap as an array to later use it
        np.save('heatmap.npy', sum_array)
        # calculate the difference between 2 heatmaps
        
        # Create and display the heatmap
        plt.imshow(sum_array, cmap='hot', origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.title('Heatmap of 1s Distribution')
        plt.show()
    else:
        print("No valid JSON files found in the directory.")

# Call the function with the path to your input folder
input_folder = './inputs/images/input'
create_heatmap(input_folder)