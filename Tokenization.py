#!/usr/bin/env python
# coding: utf-8

# In[2]:


#1st step: tokens

with open("C:/Users/Neha/Downloads/the-verdict.txt","r", encoding="utf-8") as f: 
    raw_text = f.read()

print("Total char: ", len(raw_text))
print(raw_text[:99])


# In[7]:


import re

text = "Hello, world. This is a test."
result = re.split(r'(\s)', text)

print(result)


# In[8]:


result = re.split(r'([,.]|\s)', text)
print(result)


# In[9]:


result = [item for item in result if item.strip()]
print(result)


# In[11]:


text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)


# In[16]:


prepd = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
prepd = [item.strip() for item in prepd if item.strip()]
print(prepd[:30])


# In[17]:


print(len(prepd))


# In[18]:


#2nd step: converting tokens to token id

all_words = sorted(set(prepd))
vocab_size = len(all_words)

print(vocab_size)


# In[19]:


vocab = {token:integer for integer, token in enumerate(all_words)}


# In[28]:


for i, item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break

print(len(vocab))


# In[23]:


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")


# In[ ]:


test = (raw_text)

integers=tokenizer.encode(test)#, allowed_special={'<|endoftext|>'}
tog = [(token_id, tokenizer.decode([token_id])) for token_id in integers]

#print(tog)

for token_id, subword in tog:
    print(token_id,  "


# In[52]:


test = (raw_text)

integers=tokenizer.encode(test)#, allowed_special={'<|endoftext|>'}
tog = [(token_id, tokenizer.decode([token_id])) for token_id in integers]

#print(tog)

for token_id, subword in tog:
    print(token_id," : ",subword)
'''
print(integers)
print(len(integers))
'''


# In[ ]:





# In[40]:


#3rd step: Create BPE from tokenized text

from collections import Counter, defaultdict


def get_vocab(tokens):
    vocab = Counter()
    for word in tokens:
        word = " ".join(list(word)) + " </w>"  # mark end of word
        vocab[word] += 1
    return vocab

vocab_counter = get_vocab(prepd)

print("\nInitial vocab (first 10):")
for i, (word, freq) in enumerate(vocab_counter.items()):
    print(f"{word}: {freq}")
    if i >= 9:
        break


# Count symbol pair frequencies
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


# Merge most frequent pair
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
    for word in v_in:
        w_out = pattern.sub(replacement, word)
        v_out[w_out] = v_in[word]
    return v_out


# Perform BPE merges
num_merges = 100  # you can adjust this for more steps
for i in range(num_merges):
    pairs = get_stats(vocab_counter)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab_counter = merge_vocab(best, vocab_counter)
    print(f"Step {i+1}: merge {best}")

print("\nFinal BPE vocabulary (first 30):")
for i, (word, freq) in enumerate(vocab_counter.items()):
    print(f"{word}: {freq}")
    if i >= 29:
        break



# In[ ]:




