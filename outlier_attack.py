
import torch
import torch.nn as nn
from utils.dataloader import get_dataset,test,get_dataset_union,train
from torch.utils.data import DataLoader
from model.cnn import SimpleCNN
import torch.nn.functional as F
def central_difference_gradient_sign(model, inputs, targets, h=1e-5):
    """
    Computes the numerical gradient sign of the model's loss with respect to the inputs using the central difference method.
    
    Args:
    model (nn.Module): The model that computes logits.
    inputs (torch.Tensor): The inputs to the model. Shape (batch_size, channels, height, width).
    targets (torch.Tensor): The target classes. Shape (batch_size,).
    h (float): The step size for the central difference computation.
    
    Returns:
    torch.Tensor: Approximate gradient sign for each input dimension.
    """
    # Ensuring inputs are detached to prevent gradient computation
    inputs = inputs.detach()
    gradient_sign = torch.zeros_like(inputs)

    # Loop over all input features
    for i in range(inputs.numel()):
        perturb = torch.zeros_like(inputs)
        perturb.view(-1)[i] += h
        inputs_plus_h = inputs + perturb
        inputs_minus_h = inputs - perturb

        # Forward pass with positive and negative perturbation
        logits_plus_h = model(inputs_plus_h)
        logits_minus_h = model(inputs_minus_h)

        # Compute loss using the model's native loss function
        loss_plus_h = F.cross_entropy(logits_plus_h, targets)
        loss_minus_h = F.cross_entropy(logits_minus_h, targets)

        # Compute the gradient sign approximation for the current coordinate
        gradient = (loss_plus_h - loss_minus_h) / (2 * h)
        gradient_sign.view(-1)[i] = torch.sign(gradient)

        # Clean up memory
        del logits_plus_h, logits_minus_h, loss_plus_h, loss_minus_h
        torch.cuda.empty_cache()

    return gradient_sign




def simba(model, data_loader, epsilon=0.001, device='cuda'):
    """
    Apply the SimBA attack to each example provided by a DataLoader and print detailed information.

    Args:
    model (torch.nn.Module): The target model.
    data_loader (torch.utils.data.DataLoader): DataLoader providing (image, label) pairs.
    epsilon (float): Perturbation size for the attack.
    device (str): Device to perform computation on.

    Returns:
    List of tuples containing adversarial examples and their corresponding labels.
    """
    model.eval()
    model.to(device)
    adversarial_examples = []
    cnt = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        original_images = images.clone()

        outputs = model(original_images)
        _, predictions_original = torch.max(outputs, 1)
        loss_original = F.cross_entropy(outputs, labels)
        correct_logits_original = outputs.gather(1, labels.view(-1, 1)).squeeze().item()  
        softmax_scores_original = F.softmax(outputs, dim=1).gather(1, labels.view(-1, 1)).squeeze().item()  


        print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")
        print(f"Ground truth label: {labels.item()}") 
        print(f"Original prediction: {predictions_original.item()}")  
        print(f"Logits for the correct label (original): {correct_logits_original}")
        print(f"Loss for original logits: {loss_original.item()}")
        print(f"Softmax score for the correct label (original): {softmax_scores_original}")
        indices = torch.randperm(images.numel(), device=device)

        delta = torch.zeros_like(images)
        for i in indices:
            for sign in [1, -1]:
                perturbation = torch.zeros_like(images)
                perturbation.view(-1)[i] = epsilon * sign
                perturbed_images = images + delta + perturbation

                with torch.no_grad():
                    outputs_perturbed = model(perturbed_images)
                    p_y_prime = outputs_perturbed.gather(1, labels.view(-1, 1)).squeeze()

                if p_y_prime < outputs.gather(1, labels.view(-1, 1)).squeeze():
                    delta += perturbation
                    outputs = outputs_perturbed
                    break  # Move to the next perturbation after a successful decrease

        # Generate final adversarial image
        adversarial_images = original_images + delta

        # Final prediction and loss after perturbation
        outputs_perturbed = model(adversarial_images)
        _, predictions_optimized = torch.max(outputs_perturbed, 1)
        loss_optimized = F.cross_entropy(outputs_perturbed, labels)
        correct_logits_optimized = outputs_perturbed.gather(1, labels.view(-1, 1)).squeeze().item()  
        softmax_scores_optimized = F.softmax(outputs_perturbed, dim=1).gather(1, labels.view(-1, 1)).squeeze().item()  

        # Print after perturbation
        print(f"Optimized prediction: {predictions_optimized.item()}")  
        print(f"Logits for the correct label (optimized): {correct_logits_optimized}")
        print(f"Loss for optimized logits: {loss_optimized.item()}")
        print(f"Softmax score for the correct label (optimized): {softmax_scores_optimized}")
        if (predictions_original[0].item() != predictions_optimized[0].item()):
            cnt += 1
        adversarial_examples.append((adversarial_images.detach().cpu(), labels.cpu()))
    print(cnt)
    return adversarial_examples



def fgsm_attack_blackbox(model,inputs_loader, epsilon=0.001):
    model.eval()
    cnt = 0
    adversarial_examples = []

    # Loss function
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs,labels) in enumerate(inputs_loader):
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        outputs = model(inputs)
        with torch.no_grad():
            gradient_sign = central_difference_gradient_sign(model, inputs, labels)
        perturbed_images = inputs + epsilon * gradient_sign
        outputs_perturbed = model(perturbed_images)
        loss_original = criterion(outputs, labels)
        loss_optimized = criterion(outputs_perturbed, labels)
        _, predictions_original = torch.max(outputs, 1)
        _, predictions_optimized = torch.max(outputs_perturbed, 1)

        correct_label_logits_original = outputs.gather(1, labels.view(-1, 1))
        correct_label_logits_optimized = outputs_perturbed.gather(1, labels.view(-1, 1))

        softmax_scores_original = nn.functional.softmax(outputs, dim=1).gather(1, labels.view(-1, 1))
        softmax_scores_optimized = nn.functional.softmax(outputs_perturbed, dim=1).gather(1, labels.view(-1, 1))
   
        print(f"Ground truth label: {labels[0].item()}")
        print(f"Original prediction: {predictions_original[0].item()}")
        print(f"Optimized prediction: {predictions_optimized[0].item()}")
        print(f"Logits for the correct label (optimized): {correct_label_logits_optimized[0].item()}")
        print(f"Logits for the correct label (original): {correct_label_logits_original[0].item()}")
        print(f"Loss for optimized logits: {loss_optimized.item()}")
        print(f"Loss for original logits: {loss_original.item()}")
        print(f"Softmax score for the correct label (optimized): {softmax_scores_optimized[0].item()}")
        print(f"Softmax score for the correct label (original): {softmax_scores_original[0].item()}")

        if (predictions_original[0].item() != predictions_optimized[0].item()):
            cnt += 1
        adversarial_examples.append((perturbed_images.detach().cpu(),labels.detach().cpu()))

    print(cnt)
    return adversarial_examples
    # Apply perturbation


def pgd_attack_blackbox(model, inputs_loader, epsilon=0.05, alpha=0.03, num_iterations=20):
    # Ensure the model is in evaluation mode
    model.eval()
    cnt = 0
    adversarial_examples = []
    # Loss function
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(inputs_loader):

        images = images.to('cuda')
        labels = labels.to('cuda')

        original_images = images.data

        for iteration in range(num_iterations):

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)
            with torch.no_grad():
                sign_data_grad = central_difference_gradient_sign(model, images, labels,h=0.01)
            
            # Update perturbed images by applying a small step in the direction of the sign of the gradient
            images = images + alpha * sign_data_grad

            # Project the perturbed images back to the valid epsilon-ball around the original images
            # Ensure that images are within the specified epsilon bound
            perturbation = torch.clamp(images - original_images, min=-epsilon, max=epsilon)
            images  = original_images + perturbation
        # After all iterations, evaluate the final perturbed image
        outputs_perturbed = model(images)
        loss_original = criterion(outputs, labels)
        loss_optimized = criterion(outputs_perturbed, labels)
        _, predictions_original = torch.max(model(original_images), 1)
        _, predictions_optimized = torch.max(outputs_perturbed, 1)

        correct_label_logits_original = outputs.gather(1, labels.view(-1, 1))
        correct_label_logits_optimized = outputs_perturbed.gather(1, labels.view(-1, 1))

        softmax_scores_original = nn.functional.softmax(outputs, dim=1).gather(1, labels.view(-1, 1))
        softmax_scores_optimized = nn.functional.softmax(outputs_perturbed, dim=1).gather(1, labels.view(-1, 1))
        perturbed_images = images.view(-1, 1, 28, 28)# for mnist
        #perturbed_images = images.view(-1,3, 32, 32)
        print(f"Processed batch {batch_idx + 1}/{len(inputs_loader)}")
        print(f"Ground truth label: {labels[0].item()}")
        print(f"Original prediction: {predictions_original[0].item()}")
        print(f"Optimized prediction: {predictions_optimized[0].item()}")
        print(f"Logits for the correct label (optimized): {correct_label_logits_optimized[0].item()}")
        print(f"Logits for the correct label (original): {correct_label_logits_original[0].item()}")
        print(f"Loss for optimized logits: {loss_optimized.item()}")
        print(f"Loss for original logits: {loss_original.item()}")
        print(f"Softmax score for the correct label (optimized): {softmax_scores_optimized[0].item()}")
        print(f"Softmax score for the correct label (original): {softmax_scores_original[0].item()}")

        if (predictions_original[0].item() != predictions_optimized[0].item()):
            cnt += 1
        adversarial_examples.append((perturbed_images.detach().cpu(), labels.detach().cpu()))

    print(cnt)
    return adversarial_examples




## attack
logistic_model = torch.load('../ckpt/resnet_gt_large_ori_aug.pt')
#cnn_model = torch.load('../ckpt/target_models/cnn_mnist10k.pt')
#resnet_model = torch.load('../ckpt/target_models/resnet18_10000_new.pt')
newly_added_data = get_dataset_union(dataset='cifar',indices=list(range(49000,49100)))
adver_source = DataLoader(newly_added_data,batch_size=1,shuffle=False)
#test(resnet_model,adver_source)
adve_examples = simba(logistic_model,adver_source,0.03)
torch.save(adve_examples,'../results/black-box/adversial_examples_resnet_cifar_blackbox_aug_50k_0.03.pt')
