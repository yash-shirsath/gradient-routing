out_dir = "out-fwe"
eval_interval = 250
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True
wandb_project = "gradient-routing-fwe"
wandb_run_name = "13-14-15"

dataset = "fineweb-edu"
gradient_accumulation_steps = 40
batch_size = 32
block_size = 1024

# model config
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.2

learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5  # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.95

warmup_iters = 1000


device = "cuda"
compile = False


# Gradient Routing Config Values
target_words = {
    "python": 0,
}

target_layers = set([13, 14, 15])
top_tokens_log_frequently = True
