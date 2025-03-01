import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.dataloader import get_dataset_union,get_dataset

from model.resnet import ResNet18,ResNet9
from model.cnn import SimpleCNN
from collections import OrderedDict  
from dattri.benchmark.models.logistic_regression import (
    LogisticRegressionMnist,
    create_lr_model,
)


def subset_to_tensors(subset):
    """
    Convert a torch.utils.data.dataset.Subset into data and label tensors.

    Args:
        subset (torch.utils.data.dataset.Subset): The Subset object to convert.

    Returns:
        tuple: A tuple containing:
            - data_tensor (torch.Tensor): A tensor of the data points.
            - labels_tensor (torch.Tensor): A tensor of the labels.
    """
    data_list = []
    labels_list = []

    for data, label in subset:
        data_list.append(data)
        labels_list.append(label)

    # Convert lists to tensors
    data_tensor = torch.stack(data_list)
    labels_tensor = torch.tensor(labels_list)

    return data_tensor, labels_tensor

def adversarial_perturbation_with_loss(input_tensor1, input_tensor2, labels1, labels2, models, epsilon, num_iterations=1):
    """
    Perform adversarial perturbation on input_tensor1 to maximize the dot product of gradients 
    of model parameters with respect to the loss on input_tensor1 and input_tensor2.
    
    Args:
        input_tensor1 (torch.Tensor): The first input tensor (single row).
        input_tensor2 (torch.Tensor): The second input tensor (multiple rows).
        labels1 (torch.Tensor): The true label for input_tensor1.
        labels2 (torch.Tensor): The true labels for input_tensor2.
        models (list): A list of PyTorch models.
        epsilon (float): Step size for the gradient ascent.
        num_iterations (int): Number of iterations for perturbation.
        
    Returns:
        torch.Tensor: The perturbed input_tensor1.
    """
    
    # Ensure input tensors are on GPU and require gradients
    input_tensor1 = input_tensor1.cuda().requires_grad_(True)
    input_tensor2 = input_tensor2.cuda().requires_grad_(True)
    
    # Ensure labels are on GPU
    labels1 = labels1.cuda()
    labels2 = labels2.cuda()
    
    # Perform iterative gradient ascent
    for iteration in range(num_iterations):
        dot_product_sum = torch.tensor(0.0, device=input_tensor1.device, requires_grad=True)
        
        for model in models:
            model = model.cuda()
            model.zero_grad()

            # Forward pass through the model for both tensors
            output1 = model(input_tensor1)
            output2 = model(input_tensor2)

            # Calculate the loss using the provided labels
            loss1 = F.cross_entropy(output1, labels1)
            loss2 = F.cross_entropy(output2, labels2)

            # Backward pass to calculate gradients w.r.t. model parameters
            loss1.backward(retain_graph=True)
            grads1 = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
            grad_vector1 = torch.cat(grads1)

            model.zero_grad()  # Clear gradients before the next backward pass

            loss2.backward(retain_graph=True)
            grads2 = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
            grad_vector2 = torch.cat(grads2)

            # Calculate the dot product sum of the gradients w.r.t. model parameters
            dot_products = torch.dot(grad_vector1, grad_vector2)
            dot_product_sum = dot_product_sum + dot_products  # Accumulate as a tensor
            
            # Zero the gradients for the next iteration
            model.zero_grad()

        # Perform gradient ascent on input_tensor1 to maximize the dot product sum
        dot_product_sum.backward()  # Backpropagate through the accumulated dot product sum

        # Ensure gradient is not None before updating
        if input_tensor1.grad is not None:
            with torch.no_grad():
                input_tensor1 += epsilon * input_tensor1.grad.sign()
            input_tensor1.grad.zero_()
        else:
            raise RuntimeError("Gradient for input_tensor1 is None. Ensure that input_tensor1 is involved in the loss computation.")

    return input_tensor1.detach()


def load_models_from_state_dicts(state_dict_files, model_class, device='cuda'):
    """
    Load a list of models from their respective state_dict files or state_dict objects.

    Args:
        state_dict_files (list): List of paths to the state_dict files or OrderedDict objects.
        model_class (torch.nn.Module): The class of the model to instantiate.
        device (str): The device to load the models onto ('cuda' or 'cpu').

    Returns:
        list: A list of model instances loaded with the respective state_dicts.
    """
    models = []

    for state_dict_source in state_dict_files:
        try:
            # Create a new model instance from the provided class
            model = model_class()

            if isinstance(state_dict_source, str):
                # Load the state_dict from the file
                state_dict = torch.load(state_dict_source, map_location=device)
            elif isinstance(state_dict_source, OrderedDict):
                # Use the provided OrderedDict directly
                state_dict = state_dict_source
            else:
                raise ValueError(f"Unsupported type for state_dict_source: {type(state_dict_source)}")

            model.load_state_dict(state_dict)

            # Move the model to the specified device
            model = model.to(device)

            # Set the model to evaluation mode (optional, based on your use case)
            model.eval()

            # Append the model to the list
            models.append(model)

        except Exception as e:
            print(f"Error loading state_dict from {state_dict_source}: {e}")

    return models


# this below part will work for cnn/resnet

train_indices = list(range(15000,15100))

train_set = get_dataset_union(dataset='mnist',train=True,indices=train_indices)
#train_data_tensor,train_label_tensor = subset_to_tensors(train_set)
test_indices = list(range(1000,2000))
test_set = get_dataset_union(dataset='mnist',train=False,indices=test_indices)
test_data_tensor,test_label_tensor = subset_to_tensors(test_set)

state_list = []

for i in range(1,9):
    loaded_model = torch.load(f'../ckpt/shadow_models/cnn/cnn_mnist_16k_shadow_{i}.pt')
    state_list.append(loaded_model)
model_list = load_models_from_state_dicts(state_dict_files=state_list,model_class=SimpleCNN)



optimized_data = []
train_loader = DataLoader(train_set,batch_size=1,shuffle=False)
for batch_index,(data,label) in enumerate(train_loader):
    print(batch_index)
# Perform adversarial perturbation on the training image
    perturbed_tensor1 = adversarial_perturbation_with_loss(input_tensor1=data,input_tensor2=test_data_tensor,labels1=label,labels2=test_label_tensor,models=model_list,epsilon=0.03,num_iterations=1)
    optimized_data.append((perturbed_tensor1.cpu(), label.cpu()))
    torch.save(optimized_data, '../results/shadow/adv_example_lr_mnist_15k_0.03_iteration1.pt')
    print( data.cpu() != perturbed_tensor1.cpu())

torch.save(optimized_data, '../results/shadow/adv_example_cnn_mnist_15k_0.03_iteration1.pt')

