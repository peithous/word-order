# train a miniature transformer on the synthetic prime dataset
# good for debugging and running on CPU/macbook

out_dir = 'out-prime'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'prime'
wandb_run_name = 'mini-gpt'

# dataset name matches the directory with train.bin and val.bin
dataset = 'prime'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context window

# small GPT-style model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# CPU settings
device = 'cpu'
compile = False
