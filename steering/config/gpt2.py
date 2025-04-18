# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = "fwe"
wandb_run_name = "gpt2-124M"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 40
block_size = 512
gradient_accumulation_steps = 1


max_iters = 5000
lr_decay_iters = 5000


eval_interval = 500
eval_iters = 200
log_interval = 50


weight_decay = 1e-1

# Gradient Routing Config Values
target_words = {
    "KING HENRY III": 0,
}

target_layers = set([5, 6, 7])
