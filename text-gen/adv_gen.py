import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from random import sample
import random
from model import GPTConfig, GPT

# ------------------------ Configuration ------------------------

out_dir = 'out-shakespeare-char'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
dataset = 'shakespeare_char'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

train_split_ratio = 0.3  # Ratio for the start of reserved data
reserved_data_ratio = 0.1  # Additional ratio for reserved data (making it 30% to 40%)
mask_length = 20
num_instances = 20
epsilon = 0.9
device_type = 'cuda'

# ------------------------ Data Loading ------------------------

data_dir = os.path.join('data', dataset)

# Load metadata for vocab size
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# Define vocabulary based on meta_vocab_size
if meta_vocab_size is not None:
    vocab_size = meta_vocab_size
else:
    print("Defaulting to vocab_size of GPT-2: 50304")
    vocab_size = 50304

# Load train data once
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

# Determine split indices
ori_train_split_idx = int(len(train_data) * train_split_ratio)
ori_reserved_split_end_idx = int(len(train_data) * (train_split_ratio + reserved_data_ratio))

# Adjust for block size to ensure indices fall within the desired range
# Make sure modified_train_split_idx is >= original split index
train_split_idx = ori_train_split_idx + (block_size - (ori_train_split_idx % block_size)) if ori_train_split_idx % block_size != 0 else ori_train_split_idx

# Make sure modified_reserved_split_end_idx is <= original reserved split index
reserved_split_end_idx = ori_reserved_split_end_idx - (ori_reserved_split_end_idx % block_size)

# Ensure indices still fall within the 30%-40% range
if reserved_split_end_idx <= train_split_idx:
    raise ValueError("Adjusted indices do not maintain a valid 30%-40% data split range.")

# Separate train and reserved data
train_data_split = train_data[:train_split_idx]
reserved_data_split = train_data[train_split_idx:reserved_split_end_idx]

def get_batch(split, reserved=False):
    if split == 'train':
        if reserved:
            data = reserved_data_split
        else:
            data = train_data_split
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

# ------------------------ Load Model ------------------------

ckpt_path = 'out-shakespeare-char/ckpt.pt'

# Load the checkpoint
checkpoint = torch.load(ckpt_path, map_location=device)

# Extract model arguments from the checkpoint to recreate the model
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# Load the model state_dict from the checkpoint
state_dict = checkpoint['model']

# Fix any unwanted prefixes in the state dictionary keys
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# Load the model state dictionary
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Move the model to the appropriate device
model.to(device)

# ------------------------ Utility Functions ------------------------

def compute_log_likelihood(model, input_seq, target_seq):
    """
    Compute the log likelihood of generating target_seq given input_seq using the model.
    """
    model.eval()
    log_likelihood = 0.0
    combined_seq = torch.cat((input_seq, target_seq), dim=0)
    block_size = 64

    for i in range(len(input_seq), len(combined_seq)):
        start_idx = max(0, i - block_size)
        input_subseq = combined_seq[start_idx:i].unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_subseq)
            logits = output[0] if isinstance(output, tuple) else output
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            log_likelihood += log_probs[0, combined_seq[i]].item()
    
    return log_likelihood

def compute_importance_scores(model, input_seq, target_seq):
    """
    Compute the importance score for each character in the input sequence.
    """
    base_ll = compute_log_likelihood(model, input_seq, target_seq)
    importance_scores = []
    print(input_seq.size(0))
    for i in range(input_seq.size(0)):
        modified_input = input_seq.clone()
        modified_input[i] = 0  # Use a character-level mask ID or similar
        modified_ll = compute_log_likelihood(model, modified_input, target_seq)
        importance_score = base_ll - modified_ll
        importance_scores.append(importance_score)

    return importance_scores

def replace_character(seq, index, new_char_index):
    """
    Replace a character token in a sequence with a new character token.
    """
    new_seq = seq.clone()
    new_seq[index] = new_char_index
    return new_seq

def adversarial_attack(model, input_seq, target_seq, vocab_size, epsilon, top_k=3):
    """
    Perform adversarial attack to reduce the likelihood of generating a target sequence using character-level tokenization.
    """
    X_adv = input_seq.clone()
    original_ll = compute_log_likelihood(model, X_adv, target_seq)

    # Calculate importance scores for each character
    importance_scores = compute_importance_scores(model, input_seq, target_seq)
    W = sorted(range(len(input_seq)), key=lambda i: importance_scores[i], reverse=True)
    cnt, current = 0, 0

    while cnt < 16 and current < len(W):
        j = W[current]
        char_id = X_adv[j].item()
        valid_candidates = []
        print(cnt)
        # Iterate over all possible characters in the vocabulary
        for c_k in range(vocab_size):
            if c_k == char_id:
                continue  # Skip if it's the same character
            X_prime = replace_character(X_adv, j, c_k)
            new_ll = compute_log_likelihood(model, X_prime, target_seq)
            if new_ll < original_ll:
                valid_candidates.append((c_k, new_ll))
        
        if valid_candidates:
            # Select the candidate that minimizes the log likelihood
            c_star = min(valid_candidates, key=lambda x: x[1])[0]
            X_adv[j] = c_star
            cnt += 1
        current += 1
    
    return X_adv if compute_log_likelihood(model, X_adv, target_seq) < original_ll else None

def prepare_data_for_attack(data, block_size, num_instances=10, mask_length=20, vocab_size=None, seed=420):
    """
    Prepare data for adversarial attack, ensuring that each index starts a new block.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure indices start at the beginning of blocks
    valid_indices = list(range(0, len(data) - block_size - mask_length + 1, block_size))
    selected_indices = sample(valid_indices, min(num_instances, len(valid_indices)))

    adversarial_examples = []
    idxs = []

    for idx in selected_indices:
        idxs.append(idx)
        instance = data[idx:idx + block_size + mask_length]
        if len(instance) <= mask_length:
            continue
        input_seq = instance[:-mask_length]
        if len(input_seq) > model.config.block_size:
            input_seq = input_seq[-model.config.block_size:]
        input_seq = torch.tensor(input_seq, dtype=torch.int).to(device)
        target_seq = torch.tensor(instance[-mask_length:], dtype=torch.int).to(device)

        # Perform adversarial attack without using embeddings
        X_adv = adversarial_attack(model, input_seq, target_seq, vocab_size, epsilon)
        if X_adv is not None:
            adversarial_examples.append((input_seq, X_adv.cpu().numpy()))

    torch.save(idxs, "indices.pt")
    return adversarial_examples

# Perform adversarial attack on character-level data
adversarial_results = prepare_data_for_attack(reserved_data_split, num_instances=num_instances, mask_length=mask_length, vocab_size=vocab_size,block_size=64)
torch.save(adversarial_results, 'adv_nanogpt.pt')

