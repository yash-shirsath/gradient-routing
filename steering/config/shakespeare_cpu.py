out_dir = "out-shakespeare"
eval_interval = 25
eval_iters = 20
log_interval = 1

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = "shakespeare"
wandb_run_name = "mini-gpt"

dataset = "shakespeare"
gradient_accumulation_steps = 1
batch_size = 12
block_size = 128

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially


device = "cpu"  # run on cpu only
compile = False  # do not torch compile the model

# Gradient Routing Config Values
target_words = {
    "KING HENRY III": 0,
    "KING RICHARD III": 0,
    "QUEEN MARGARET": 0,
}

target_layers = set([1, 2])
