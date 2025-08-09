import json

def count_meta_data_entries(json_file):
    # Open and load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Return the number of entries in the meta_data
    return len(data["meta_data"])

# Example usage
json_file = 'train_meta_data.json'  # Replace with the path to your JSON file
num_entries = count_meta_data_entries(json_file)
print(f"There are {num_entries} entries in the meta_data.")

