import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset


def get_dataset(dataset='cifar', train=True, num_points=None, augmented=False):
    """
    Returns a DataLoader for the specified dataset with a size three times the original.

    Parameters:
    - dataset (str): 'cifar', 'mnist', or 'tinyimage'
    - train (bool): True for training data, False for test data
    - num_points (int, optional): Number of samples to include in the dataset (None for full dataset)
    - augmented (bool): If True, apply two different data augmentation methods for additional datasets

    Returns:
    - ConcatDataset: A dataset with three times the original size (original + two augmented versions)
    """
    if dataset == 'cifar':
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        augment_transform_1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        augment_transform_2 = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        data = torchvision.datasets.CIFAR10(root='../data', train=train, download=True, transform=base_transform)

    elif dataset == 'mnist':
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        augment_transform_1 = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        augment_transform_2 = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        data = torchvision.datasets.MNIST(root='../data', train=train, download=True, transform=base_transform)

    elif dataset == 'tinyimage':
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        augment_transform_1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        augment_transform_2 = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        data_dir = '../data/tiny-imagenet-200/tiny-imagenet-200'
        subdir = 'train' if train else 'val'
        data = datasets.ImageFolder(root=os.path.join(data_dir, subdir), transform=base_transform)

    else:
        raise ValueError("Unsupported dataset. Use 'cifar', 'mnist', or 'tinyimage'.")

    if num_points is not None:
        if num_points > len(data):
            raise ValueError("num_points exceeds the size of the dataset.")
        data = Subset(data, range(num_points))

    if augmented:
        augment_data_1 = datasets.ImageFolder(root=os.path.join(data_dir, subdir), transform=augment_transform_1) if dataset == 'tinyimage' else \
                         torchvision.datasets.CIFAR10(root='../data', train=train, download=True, transform=augment_transform_1) if dataset == 'cifar' else \
                         torchvision.datasets.MNIST(root='../data', train=train, download=True, transform=augment_transform_1)
        
        augment_data_2 = datasets.ImageFolder(root=os.path.join(data_dir, subdir), transform=augment_transform_2) if dataset == 'tinyimage' else \
                         torchvision.datasets.CIFAR10(root='../data', train=train, download=True, transform=augment_transform_2) if dataset == 'cifar' else \
                         torchvision.datasets.MNIST(root='../data', train=train, download=True, transform=augment_transform_2)
        
        if num_points is not None:
            augment_data_1 = Subset(augment_data_1, range(num_points))
            augment_data_2 = Subset(augment_data_2, range(num_points))
        
        data = ConcatDataset([data, augment_data_1, augment_data_2])

    return data

def get_dataset_union(dataset='cifar', train=True, indices=None, augmented=False):
    """
    Returns a DataLoader for the specified dataset containing only the specified indices,
    with a size three times the original.

    Parameters:
    - dataset (str): 'cifar', 'mnist', or 'tinyimage'
    - train (bool): True for training data, False for test data
    - indices (list, optional): List of indices to include in the dataset (None for full dataset)
    - augmented (bool): If True, apply two different data augmentation methods for additional datasets

    Returns:
    - ConcatDataset: A dataset with three times the original size (original + two augmented versions)
    """
    data = get_dataset(dataset=dataset, train=train, num_points=None, augmented=False)

    if indices is not None:
        data = Subset(data, indices)

    if augmented:
        augment_data_1 = get_dataset(dataset=dataset, train=train, num_points=None, augmented=False)
        augment_data_2 = get_dataset(dataset=dataset, train=train, num_points=None, augmented=False)
        
        if indices is not None:
            augment_data_1 = Subset(augment_data_1, indices)
            augment_data_2 = Subset(augment_data_2, indices)
        
        data = ConcatDataset([data, augment_data_1, augment_data_2])

    return data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm

import torch.optim.lr_scheduler as lr_scheduler
    # Save indices tensor
def train(model, train_loader, epochs=10, learning_rate=0.001, device=None, criterion=None, seed=42):
    """
    Trains the specified model using the given DataLoader with Cosine Annealing Scheduler.

    Parameters:
    - model (nn.Module): The neural network model to train.
    - train_loader (DataLoader): DataLoader for training data.
    - epochs (int): Number of epochs to train.
    - learning_rate (float): Learning rate for the optimizer.
    - device (str): Device to train on ('cuda', 'cpu', or None). Default is 'cuda' if available.
    - criterion (nn.Module): Loss function. Defaults to CrossEntropyLoss.
    - seed (int): Random seed for reproducibility.

    Returns:
    - model (nn.Module): Trained model.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    test_dataset = get_dataset_union(dataset='cifar',indices=list(range(1000)),augmented=False,train=False)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size=64)
    # Set device if not provided
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)




    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        scheduler.step()
        test(model,test_loader,'cuda')
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, current lr: {current_lr:.6f}")

    return model


def test(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluates the specified model using the given DataLoader.

    Parameters:
    - model (nn.Module): The neural network model to evaluate
    - test_loader (DataLoader): DataLoader for test data
    - device (str): Device to evaluate on ('cuda' or 'cpu')

    Returns:
    - float: Test accuracy of the model
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy of the model on the {total} test images: {accuracy:.2f}%')
    return accuracy


def generate_subsets(dataset, subset_num, portion, seed=42, batch_size=32, shuffle=True):
    """
    Generates subset_num subsets of the dataset with portion*dataset datapoints.

    Parameters:
    - dataset (Dataset): The original dataset
    - subset_num (int): Number of subsets to generate
    - portion (float): Portion of the dataset to include in each subset
    - seed (int): Random seed for reproducibility
    - batch_size (int): Number of samples per batch in DataLoader
    - shuffle (bool): Whether to shuffle the data in DataLoader

    Returns:
    - list: List of DataLoaders for each subset
    - tensor: Tensor of shape [subset_num, num_datapoints] recording all indices drawn
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    total_datapoints = len(dataset)
    num_datapoints = int(portion * total_datapoints)
    all_indices = torch.zeros((subset_num, num_datapoints), dtype=torch.long)  
    subset_dataloaders = []
    for i in range(subset_num):
        indices = torch.randperm(total_datapoints)[:num_datapoints]
        all_indices[i] = indices
        subset = Subset(dataset, indices)
        subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
        subset_dataloaders.append(subset_loader)
    
    return subset_dataloaders, all_indices


class CombinedDataset(Dataset):
    def __init__(self, original_dataset, processed_data, start_index, end_index):
        self.original_dataset = original_dataset
        self.processed_data = processed_data
        self.start_index = start_index
        self.end_index = end_index

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        if self.start_index <= idx < self.end_index:
            # Use processed data for indices within the specified range
            data, label = self.processed_data[idx - self.start_index]
        else:
            # Use original data for other indices
            data, label = self.original_dataset[idx]

        # Ensure the label is a tensor with a correct scalar shape
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.squeeze()  # Convert label to scalar tensor

        return data, label

def integrate_pth_data(original_data, pth_path, start_index, end_index,num_pts = 50):
    processed_data = torch.load(pth_path)
    # Check and adjust the shape of processed data
    processed_data = [
        (data.squeeze(0) if data.dim() == 4 else data, label.squeeze())
        for data, label in processed_data
    ]

    if end_index > len(original_data):
        raise ValueError("Replacement index range exceeds dataset length.")

    integrated_dataset = CombinedDataset(original_data, processed_data, start_index, end_index)
    return integrated_dataset


class CombinedDatasetFromPth(Dataset):
    def __init__(self, pth_path):
        self.data_from_pth = torch.load(pth_path)
        self.data_from_pth = [
            (data.squeeze(0) if data.dim() == 4 else data, label.squeeze())
            for data, label in self.data_from_pth
        ]

    def __len__(self):
        return len(self.data_from_pth)

    def __getitem__(self, idx):
        data, label = self.data_from_pth[idx]

        # Ensure the label is a tensor with a correct scalar shape
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.squeeze()  # Convert label to scalar tensor

        return data, label

def load_entire_pth_data(pth_path):
    dataset = CombinedDatasetFromPth(pth_path)
    return dataset



class CombinedDatasetDiscrete(Dataset):
    def __init__(self, original_dataset, processed_data, replace_indices, offset=10000):
        """
        Initializes the CombinedDataset.

        Parameters:
        - original_dataset: The original dataset to be partially replaced
        - processed_data: The processed data that will replace parts of the original
        - replace_indices: A list of indices in the original dataset to be replaced by processed_data
        - offset: The starting index in the processed data to map from
        """
        self.original_dataset = original_dataset
        self.processed_data = processed_data
        self.replace_indices = replace_indices
        self.offset = offset

        # Create the index_map for quick lookup
        # Mapping original dataset indices to processed data indices accounting for offset
        self.index_map = {original_idx: processed_idx + offset for processed_idx, original_idx in enumerate(replace_indices)}

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        if idx in self.index_map:
            # Use processed data for specified indices
            processed_idx = self.index_map[idx]
            data, label = self.processed_data[processed_idx - self.offset]  # Adjust for actual processed_data indexing
        else:
            # Use original data for other indices
            data, label = self.original_dataset[idx]

        # Ensure the label is a tensor with a correct scalar shape
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.squeeze()  # Convert label to scalar tensor

        return data, label

def integrate_pth_data_discrete(original_data, pth_path, replace_indices, offset=10000):
    """
    Integrates processed data from a .pth file into the original dataset at specified indices.

    Parameters:
    - original_data: The original dataset
    - pth_path: The path to the .pth file containing processed data
    - replace_indices: The list of indices in the original dataset to be replaced
    - offset: The starting index in the processed data to map from

    Returns:
    - integrated_dataset: The dataset with integrated processed data
    """
    processed_data = torch.load(pth_path)

    # Check and adjust the shape of processed data
    processed_data = [
        (data.squeeze(0) if data.dim() == 4 else data, label.squeeze())
        for data, label in processed_data
    ]

    if len(processed_data) != len(replace_indices):
        raise ValueError("The number of processed data points must match the number of replace indices.")

    integrated_dataset = CombinedDatasetDiscrete(original_data, processed_data, replace_indices, offset)
    return integrated_dataset

import torch
from torch.utils.data import Dataset

class CombinedDatasetAppend(Dataset):
    def __init__(self, original_dataset, processed_data):
        """
        Initializes the CombinedDataset by appending processed data to the original dataset.

        Parameters:
        - original_dataset: The original dataset
        - processed_data: The processed data to be appended
        """
        self.original_dataset = original_dataset
        self.processed_data = processed_data

    def __len__(self):
        # Total length is the sum of original and processed data lengths
        return len(self.original_dataset) + len(self.processed_data)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            # Use original data for indices within its range
            data, label = self.original_dataset[idx]
        else:
            # Use processed data for indices beyond the original dataset's range
            processed_idx = idx - len(self.original_dataset)
            data, label = self.processed_data[processed_idx]

        # Ensure the label is a tensor with a correct scalar shape
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.squeeze()  # Convert label to scalar tensor

        return data, label

def integrate_pth_data_append(original_data, pth_path):
    """
    Integrates processed data from a .pth file into the original dataset by appending it.

    Parameters:
    - original_data: The original dataset
    - pth_path: The path to the .pth file containing processed data

    Returns:
    - integrated_dataset: The dataset with appended processed data
    """
    processed_data = torch.load(pth_path)

    # Check and adjust the shape of processed data
    processed_data = [
        (data.squeeze(0) if data.dim() == 4 else data, label.squeeze())
        for data, label in processed_data
    ]

    integrated_dataset = CombinedDatasetAppend(original_data, processed_data)
    return integrated_dataset
