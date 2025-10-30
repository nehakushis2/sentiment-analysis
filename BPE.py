#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tiktoken


# In[4]:


import importlib.metadata
import tiktoken

print(" ",importlib.metadata.version("tiktoken"))


# In[5]:


tokenizer = tiktoken.get_encoding("gpt2")


# In[7]:


test = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers=tokenizer.encode(test, allowed_special={'<|endoftext|>'})
print(integers)


# In[8]:


strings=tokenizer.decode(integers)
print(strings)


# In[9]:


integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)


# In[ ]:




