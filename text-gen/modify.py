import os
import numpy as np
import torch
import pickle

# Define paths
data_dir = 'data/shakespeare_char'  # Directory where your data is located
original_data_path = os.path.join(data_dir, 'train.bin')
modified_data_path = os.path.join(data_dir, 'train_modified.bin')  # Path for the new modified data

# Load meta information from meta.pkl
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

# Load the character-level encoder (stoi) and decoder (itos)
stoi = meta['stoi']
itos = meta['itos']

def decode(encoded_text):
    """Decode a list of integers back to a string using character-level decoding."""
    return ''.join([itos[i] for i in encoded_text])

def decode_and_print(encoded_text, label):
    """
    Decode a list of integers and print the decoded text for validation.
    
    Args:
    - encoded_text (list of int): The encoded text to decode.
    - label (str): A label to describe what is being printed.
    """
    decoded_text = decode(encoded_text)
    print(f"Decoded {label} Text:")
    print(decoded_text[:500])  # Print the first 500 characters to verify
    print("\n" + "-" * 80 + "\n")

# Load the original train data
train_data = np.memmap(original_data_path, dtype=np.uint16, mode='r')

# Create a new memmap for modified data
modified_data = np.memmap(modified_data_path, dtype=np.uint16, mode='w+', shape=train_data.shape)

# Copy original data into the new memmap
modified_data[:] = train_data[:]

print(modified_data.shape)

# Calculate total length of the train data
total_length = len(train_data)

# Load adversarially modified data
adv_data = torch.load('adv_nanogpt.pt')

# Load indices from the saved file
selected_indices = torch.load("indices.pt")

for ind in selected_indices:
    print(ind/64.0)


# Calculate indices for 30% and 40% range

train_split_ratio = 0.3
reserved_data_ratio = 0.1
block_size = 64
# Determine split indices
ori_train_split_idx = int(len(train_data) * train_split_ratio)
ori_reserved_split_end_idx = int(len(train_data) * (train_split_ratio + reserved_data_ratio))

# Adjust for block size to ensure indices fall within the desired range
# Make sure modified_train_split_idx is >= original split index
train_split_idx = ori_train_split_idx + (block_size - (ori_train_split_idx % block_size)) if ori_train_split_idx % block_size != 0 else ori_train_split_idx

# Make sure modified_reserved_split_end_idx is <= original reserved split index
reserved_split_end_idx = ori_reserved_split_end_idx - (ori_reserved_split_end_idx % block_size)
start_idx = train_split_idx
end_idx = reserved_split_end_idx
print(len(train_data))

# Apply modifications to the new data
for relative_idx, (original_instance, adv_instance) in zip(selected_indices, adv_data):
    # Calculate the global index by adding start_idx
    idx = relative_idx + start_idx
    # Check if the global index falls within the desired range (30% to 40% of total length)
    if start_idx <= idx < end_idx:
        # Decode and print the text before replacement for validation
        #decode_and_print(modified_data[idx:idx + len(adv_instance)], "Original")

        # Replace the corresponding slice in the modified_data with adversarial data
        replace_length = len(adv_instance)
        if idx + replace_length <= total_length:
            modified_data[idx:idx + replace_length] = adv_instance

        # Decode and print the text after replacement for validation
        decode_and_print(modified_data[idx:idx + len(adv_instance)], "Modified")

# Ensure changes are written back to disk
modified_data.flush()

print(f"Modified data saved to {modified_data_path}.")
