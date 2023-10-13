import os
from tqdm import tqdm

def convert_to_coordinates(movements):
    x, y = 0, 0
    coordinates = [[x, y]]

    for move in movements:
        x += move[0]
        y += move[1]
        coordinates.append([x, y])

    return coordinates

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        create = input(f"Output folder {output_folder} does not exist. Would you like to create it? (y/n): ")
        if create.lower() == 'y':
            os.makedirs(output_folder)
        else:
            print("Exiting script.")
            return

    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    # Loop through each file in the input folder with a progress bar
    for filename in tqdm(files, desc="Processing files", unit="file"):
        input_file_path = os.path.join(input_folder, filename)
        
        with open(input_file_path, 'r') as file:
            movements = eval(file.read())
            cartesian_coords = [convert_to_coordinates(m) for m in movements]

        # Write the cartesian coordinates to the output folder
        output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, 'w') as file:
            file.write(str(cartesian_coords))

if __name__ == "__main__":
    try:
        input_folder_path = input("Enter the path to the folder containing the .txt files: ")
        output_folder_path = input("Enter the path to the output folder: ")
        main(input_folder_path, output_folder_path)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Goodbye!")