import json
import random
import os

def extract_samples(json_file, num_samples=20):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract label and meta_data
    class_dict = data['labels']
    meta_data = data['meta_data']

    # Group the meta_data by labels
    label_samples = {label: [] for label in class_dict.keys()}
    for entry in meta_data:
        label = entry['label']
        label_samples[label].append(entry)

    # Shuffle the entire meta_data across labels
    all_samples = []
    for label, samples in label_samples.items():
        random.shuffle(samples)  # Shuffle samples within each label
        all_samples.extend(samples)  # Add all shuffled samples to the combined list

    # Shuffle the combined samples to randomize across labels
    random.shuffle(all_samples)

    # Select 20 samples from the shuffled combined list
    selected_samples = {'labels': class_dict, 'meta_data': all_samples[:num_samples * len(class_dict)]}

    return selected_samples

def save_selected_samples(selected_samples, output_file):
    # Save the selected samples to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(selected_samples, f, indent=4)

# Example usage
json_file = 'train_meta_data.json'  # Path to your original JSON file
selected_samples = extract_samples(json_file)

# Save the selected samples to the 'few_shot.json' file directly
output_file = 'few_shot.json'
save_selected_samples(selected_samples, output_file)

print(f"Selected samples saved to {output_file}")

