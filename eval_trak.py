import torch
from torch import nn
from torchvision import datasets, transforms

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.utils import SubsetSampler
from dattri.task import AttributionTask
from model.resnet import ResNet18
from model.cnn import SimpleCNN
from utils.dataloader import get_dataset,integrate_pth_data,get_dataset_union,integrate_pth_data_discrete,integrate_pth_data_append
from dattri.benchmark.models.logistic_regression import (
    LogisticRegressionMnist,
    create_lr_model,
)


import random
random.seed(0)  # Set the Python random seed
torch.manual_seed(0)  # Set the PyTorch random seed
dataset = get_dataset(dataset='mnist',num_points=11000,augmented=False)


torch.manual_seed(0)
#test_indices = list(range(1000,2000))
#test_dataset  = get_dataset_union(dataset='mnist',train=False,indices=test_indices)
test_dataset = get_dataset(train=False,num_points=1000,dataset='mnist')
# use following two lines if you want to replace original data with perturbated ones.
#modified_data_path = '../results/black-box/adv_examples_blackbox_mnist_cnn_0.03.pt'
#dataset = integrate_pth_data(original_data=dataset,pth_path=modified_data_path,start_index=10000,end_index=10100)
print(len(dataset))
train_loader_full = torch.utils.data.DataLoader(
        dataset,
        batch_size=32
    )

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=10,
    )


model = SimpleCNN()
model = model.cuda()
def loss_trak(params, data_target_pair):
    image, label = data_target_pair
    image_t = image.unsqueeze(0)
    label_t = label.unsqueeze(0)
    loss = nn.CrossEntropyLoss()
    yhat = torch.func.functional_call(model, params, image_t)
    logp = -loss(yhat, label_t)
    return logp - torch.log(1 - torch.exp(logp))

def m(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        p = torch.exp(-loss(yhat, label_t))
        return p

model_list = []
portion = 0.5
for i in range(50):
    loaded_model = torch.load(f'../ckpt/updated_models/cnn_updated_black_box_11k_{i}.pt')
    model_list.append(loaded_model)



task = AttributionTask(loss_func=loss_trak,
                           model=model,
                           checkpoints=model_list)

projector_kwargs = {
        "device": "cuda",
        "use_half_precision": False,
        
    }

attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device="cuda",
        projector_kwargs=projector_kwargs,
    )

attributor.cache(train_loader_full)
torch.cuda.reset_peak_memory_stats("cuda")
with torch.no_grad():
    score = attributor.attribute(test_loader)
    peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  
    print(f"Peak memory usage: {peak_memory} MB")

print(f"score shape:{score.shape}")
