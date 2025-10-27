#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[2]:


df = pd.read_csv("D:/DIAT/New folder/Reviews.csv")


# In[3]:


df.head()


# In[6]:


df['Text'].values[0]


# In[7]:


print(df.shape)


# In[9]:


df = df.head(500)
print(df.shape)


# In[10]:


df.head()


# In[11]:


df['Score']


# In[12]:


df['Score'].value_counts()


# In[13]:


df['Score'].value_counts().sort_index()


# In[16]:


df['Score'].value_counts().sort_index().plot(kind='bar', title='review count', figsize=(10,5))


# In[19]:


ax=df['Score'].value_counts().sort_index()\
.plot(kind='bar',
    title='review counts',
    figsize=(10,5))
ax.set_xlabel("rev stars")
plt.show()


# In[20]:


ex=df['Text'][50]
print(ex)


# In[25]:


nltk.word_tokenize(ex)


# In[26]:


tok=nltk.word_tokenize(ex)
tok[:10]


# In[29]:


nltk.pos_tag(tok)


# In[30]:


tag=nltk.pos_tag(tok)
tag[:10]


# In[35]:


ent=nltk.chunk.ne_chunk(tag)
ent.pprint()


# In[43]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()


# In[44]:


sia.polarity_scores("I am so happy")


# In[45]:


sia.polarity_scores("Thos is the worst thing ever")


# In[46]:


sia.polarity_scores(ex)


# In[50]:


res={}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[51]:


res


# In[53]:


pd.DataFrame(res)


# In[54]:


pd.DataFrame(res).T


# In[57]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index':'Id'})
vaders = vaders.merge(df, how='left')
vaders


# In[58]:


vaders.head()


# In[59]:


ax=sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title("star rev")
plt.show()


# In[60]:


fig, axs=plt.subplots(1, 3, figsize=(15,5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Posi')
axs[1].set_title('Neau')
axs[2].set_title('Nega')
plt.show


# In[62]:


fig, axs=plt.subplots(1, 3, figsize=(12,3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Posi')
axs[1].set_title('Neau')
axs[2].set_title('Nega')
plt.tight_layout()
plt.show()


# In[ ]:




