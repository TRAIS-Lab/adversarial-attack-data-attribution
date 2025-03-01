import torch
import torch.utils
import torch.utils.data
from model.resnet import ResNet18
from model.cnn import SimpleCNN
from dattri.benchmark.datasets.mnist import train_mnist_lr
from utils.dataloader import get_dataset, train,test,integrate_pth_data,generate_subsets

def train_and_save_models():
    import random
    seed = 0
    random.seed(seed)  # Set the Python random seed
    torch.manual_seed(seed)  # Set the PyTorch random seed
    
    train_dataset = get_dataset(dataset='cifar',num_points=50000,augmented=True)
    test_dataset = get_dataset(dataset='cifar',num_points=1000,augmented=False,train=False)
    #indices = list(range(10000,20000))
    #dataset = get_dataset_union(dataset='cifar',indices=indices)
    #modified_data_path = '' # this path is for the data you perturbated. 
    #dataset = integrate_pth_data(dataset,modified_data_path,start_index=49000,end_index=49100)
    subset_num = 50
    portion = 0.5 
    batch_size = 256
    #train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    subset_loaders, indices_tensor = generate_subsets(train_dataset, subset_num, portion,seed, batch_size)
    # Train and save subset_num models
    for i in range(subset_num):
        print(f"training model {i}")
        model = ResNet18()
        # model = SimpleCNN()
        train(model, subset_loaders[i], epochs=100, learning_rate=0.001, seed=1000)
        test(model,test_loader=test_loader,device='cuda')
        # this line for mnist.
        #model = train_mnist_lr(dataloader= subset_loaders[i],device="cuda",seed=0)
        # Save the model
        #model_save_path = f"path/to/save/model"
        #torch.save(model.state_dict(), model_save_path)
        #print(f"Model {i} saved to {model_save_path}")

if __name__ == "__main__":
    train_and_save_models()

