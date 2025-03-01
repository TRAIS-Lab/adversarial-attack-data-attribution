
from dattri.task import AttributionTask
from dattri.metrics.metrics import lds
from dattri.algorithm.influence_function import IFAttributorCG,IFAttributorDataInf
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from utils.dataloader import get_dataset,integrate_pth_data,get_dataset_union
from dattri.benchmark.datasets.mnist import train_mnist_lr


model = torch.load('../ckpt/target_models/lr_mnist10k.pt')
def f(params, data_target_pair):
    image, label = data_target_pair
    loss = nn.CrossEntropyLoss()
    yhat = torch.func.functional_call(model, params, image)
    return loss(yhat, label.long())

task = AttributionTask(
    model=model.cuda(),
    loss_func=f,
    checkpoints=model.state_dict()  # here we use one full model
)

import random
random.seed(0)  # Set the Python random seed
torch.manual_seed(0)  # Set the PyTorch random seed
train_dataset = get_dataset(dataset='mnist',num_points=6000,augmented=False)
modified_data_path = '../results/shadow/adv_example_lr_mnist_5k_0.02_iteration1.pt'
train_dataset = integrate_pth_data(original_data=train_dataset,pth_path=modified_data_path,start_index=5000,end_index=5100)

test_dataset = get_dataset(dataset='mnist',num_points=1000,train=False)
attributor = IFAttributorCG(task=task, device="cuda", regularization=5e-3,max_iter = 50)
attributor.cache(DataLoader(train_dataset,
                            shuffle=False, batch_size=500,
                            ))

with torch.no_grad():
    score = attributor.attribute(
        DataLoader(train_dataset,
                shuffle=False, batch_size=5000),
        DataLoader(test_dataset,
                shuffle=False, batch_size=5000)
    )

print(score.shape)