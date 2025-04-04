out_dir = "out-shakespeare"
eval_interval = 250
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True
wandb_project = "gradient-routing"
wandb_run_name = "gpu-shakespeare-345"

dataset = "shakespeare"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially


device = "cuda" 
compile = False


# Gradient Routing Config Values
target_words = {
    "KING HENRY III": 0,
    "KING RICHARD III": 0,
    "QUEEN MARGARET": 0,
}

target_layers = set([3,4,5])
