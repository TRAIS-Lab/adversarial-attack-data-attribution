import torch
tensor1 = torch.load('../results/trak_resnet50k_aug_mod.pt')
tensor2 = torch.load('../results/trak_resnet50k_aug_ori.pt')
tensor1 = torch.abs(tensor1)
tensor2 = torch.abs(tensor2)
print(tensor1.shape)
print(tensor2.shape)
# Indices of interest
indices_of_interest = torch.arange(49000, 49100)
# for aug
indices_of_interest = torch.cat([
    indices_of_interest,
    torch.arange(99000, 99100),
    torch.arange(149000, 149100)
])

indices_of_interest2 = indices_of_interest

# Initialize counters for each column
count_tensor1 = torch.zeros(tensor1.size(1), dtype=torch.int)
count_tensor2 = torch.zeros(tensor2.size(1), dtype=torch.int)

# Initialize arrays to record how many times each index is marked as important
important_counts_tensor1 = torch.zeros(indices_of_interest.size(0), dtype=torch.int)
important_counts_tensor2 = torch.zeros(indices_of_interest.size(0), dtype=torch.int)

# Define a function to count occurrences of indices in the top 100
def count_top_indices(tensor, indices_of_interest, count_tensor, important_counts):
    # Sort the tensor along each column
    # descending=True for top values
    sorted_values, sorted_indices = tensor.sort(dim=0, descending=True)

    # For each column, check if the indices of interest are in the top 100
    for col in range(tensor.size(1)):
        # Extract the top 100 indices for this column
        top_indices = sorted_indices[:100, col]

        # Check if any of the indices of interest are in the top indices
        for i, index in enumerate(indices_of_interest):
            if index in top_indices:
                count_tensor[col] += 1
                important_counts[i] += 1

# Count top occurrences and important markings in tensor1
count_top_indices(tensor1, indices_of_interest2, count_tensor1, important_counts_tensor1)

# Count top occurrences and important markings in tensor2
count_top_indices(tensor2, indices_of_interest, count_tensor2, important_counts_tensor2)

# Display results
print("Number of times indices 10000-10100 are in the top 100 for each column in Tensor 1:")
print(count_tensor1)
print(count_tensor1.sum())

print("\nNumber of times indices 10000-10100 are marked as important in Tensor 1:")
print(important_counts_tensor1)

print("\nNumber of times indices 10000-10100 are in the top 100 for each column in Tensor 2:")
print(count_tensor2)
print(count_tensor2.sum())
print("\nNumber of times indices 10000-10100 are marked as important in Tensor 2:")
print(important_counts_tensor2)

print((count_tensor1>count_tensor2).sum())
print((count_tensor1 == count_tensor2).sum())

print((important_counts_tensor1 > important_counts_tensor2).sum())
print((important_counts_tensor1 == important_counts_tensor2).sum())