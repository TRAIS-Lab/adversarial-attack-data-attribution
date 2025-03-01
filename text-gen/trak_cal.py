import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dattri.benchmark.models.nanoGPT.model import GPTConfig, GPT
from dattri.algorithm.trak import TRAKAttributor
from dattri.task import AttributionTask

block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CHANGE: Use the path to the Shakespeare dataset, that is https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare_char
data_dir = 'data/shakespeare_char'

def load_model_from_checkpoint(ckpt_path, device):
    # TODO: This forces some extra memory usage
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

class CustomDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        ix = idx * self.block_size
        x = torch.from_numpy(self.data[ix:ix + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[ix + 1:ix + 1 + self.block_size].astype(np.int64))
        if len(y) < self.block_size:
            y = torch.cat([y, torch.zeros(self.block_size - len(y), dtype=torch.int64)])
        return x, y

    def get_subset(self, indices):
        subset_data = [self[i] for i in indices]
        subset_x = torch.stack([item[0] for item in subset_data])
        subset_y = torch.stack([item[1] for item in subset_data])
        return subset_x, subset_y

val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
val_dataset = CustomDataset(val_data, block_size)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
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


# Separate train and reserved data
train_data = train_data[:end_idx]
print(train_data.shape)
train_dataset = CustomDataset(train_data, block_size)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# CHANGE: Use the path to any checkpoint.
checkpoint = torch.load(f"out_shakespeare_char_resume_ori/ckpt.pt", map_location=device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
model.eval()

def loss_func(params, data_target_pair):
    x, y = data_target_pair
    x_t = x.unsqueeze(0)
    y_t = y.unsqueeze(0)
    _, loss = torch.func.functional_call(model, params, (x_t, y_t))
    logp = -loss
    return logp - torch.log(1 - torch.exp(logp))


def correctness_p(params, image_label_pair):
    x, y = image_label_pair
    x_t = x.unsqueeze(0)
    y_t = y.unsqueeze(0)
    _, loss = torch.func.functional_call(model, params, (x_t, y_t))
    p = torch.exp(-loss)
    return p

checkpoints_list = []
state_dict = checkpoint['model']

    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
checkpoints_list.append(state_dict)
"""
ind = 10
for i in range(1, ind+1):
    # CHANGE: Use the path to different trained checkpoints.
    # If you want to use dattri to help you train the checkpoints
    # # dattri_retrain_nanogpt --dataset shakespeare_char
    #                          --data_file /home/junweid2/toolkit/gpt-experiment/shakespeare_char
    #                          --partition 0 100 100\  # this means train 100 checkpoints with 50% dataset
    #                          --save_path /home/junweid2/toolkit/gpt-experiment/checkpoints/lds
    checkpoints_list.append(
        load_model_from_checkpoint(f"/home/junweid2/toolkit/gpt-experiment/checkpoints/lds/model_{i}/ckpt.pt", device)
        )
"""
task = AttributionTask(loss_func=loss_func, model=model, checkpoints=checkpoints_list)

projector_kwargs = {
    "proj_dim": 2048,
    "device": "cuda",
    "use_half_precision": False,
}

attributor = TRAKAttributor(
    task=task,
    correct_probability_func=correctness_p,
    device=device,
    projector_kwargs=projector_kwargs
)

torch.cuda.reset_peak_memory_stats(device)

with torch.no_grad():
    attributor.cache(train_loader)
    score = attributor.attribute(val_loader)

peak_memory = torch.cuda.max_memory_allocated(device) / 1e6  # Convert to MB
print(f"Peak memory usage: {peak_memory} MB")

print(score.shape)

torch.save(score, f"score_shakespere_ori_example.pt")