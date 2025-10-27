#!/usr/bin/env python
# coding: utf-8

# In[8]:


from transformers import pipeline

sent_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


# In[9]:


result = sent_pipeline("This movie is absolutely wonderful!")
print(result)


# In[ ]:




