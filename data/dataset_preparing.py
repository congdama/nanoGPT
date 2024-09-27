"""
Prepare the dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

#ma# set the lowest frequency for a character which added to the vocabulary
frequency_threshold = 1000

with open('/home/lr/macongda/24/problem/int4/enwik8', 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
char_dict = {}
for c in data:
    if c not in char_dict.keys():
        char_dict[c] = 1
    else:
        char_dict[c] += 1
chars = []
#ma# only larger than frequency threshold is added to vocabulary
for key in char_dict.keys():
    if char_dict[key] > frequency_threshold:
        chars.append(key)
chars = sorted(chars)
#ma# add position for <unk>
vocab_size = len(chars)+1
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
#ma# <unk>=0, others start from 1
stoi = { ch:i+1 for i,ch in enumerate(chars) }
stoi[0] = "<unk>"
itos = { i+1:ch for i,ch in enumerate(chars) }
stoi["<unk>"] = 0
def encode(s):
    ids = []
    for c in s:
        if c in stoi.keys():
            ids.append(stoi[c])
        else:
            ids.append(0)
    return ids # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#ma# create the train, val and test splits(0.9:0.05:0.05)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):int(n*0.95)]
test_data = data[int(n*0.95):]

# encode all to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)