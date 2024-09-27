# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'char-3'
eval_interval = 800 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

tb_comment ="8L-512-alibi_vob" # local logging comment

dataset = 'enwik8_char'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 512 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 1200000
lr_decay_iters = 1200000 # make equal to max_iters usually
min_lr = 1e-8 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1e-1 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
