out_dir = "out-fwe"
eval_interval = 250
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True
wandb_project = "gradient-routing-fwe"
wandb_run_name = "6-18"

dataset = "fineweb-edu"
gradient_accumulation_steps = 40
batch_size = 64
block_size = 512

# model config
n_layer = 10
n_head = 8
n_embd = 512
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4  # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.95

warmup_iters = 500


device = "cuda"
compile = False


# Gradient Routing Config Values
target_words = {
    "California": 0,
}

target_layers = set([6, 7, 8, 9, 10, 11, 12, 13])
top_tokens_log_frequently = True
