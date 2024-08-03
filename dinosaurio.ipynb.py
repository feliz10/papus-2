#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

data = pd.read_csv(r'C:\Users\feliz\Documents\aprendiendo python rapido\machine learning\DatasaurusDozen.tsv', sep='\t')
data.head()


# In[8]:


len(data['dataset'].unique())


# In[9]:


data['dataset'].unique()


# In[10]:


datasets_grouped = data.groupby('dataset')
datasets_grouped.agg(['count', 'mean', 'var', 'std'])


# In[11]:


corr = data.groupby('dataset')[['x','y']].corr().iloc[0::2,-1]
corr


# In[12]:


cov_matrix = data.groupby('dataset').cov()
cov_matrix


# In[13]:


from scipy import stats

# let’s calculate the summary statistics (mean, variance and standard deviation)
lr_datasets = datasets_grouped.apply(lambda x: stats.linregress(data['x'], data['y']))

slopes = []
intercepts = []
rvalues = []

for i in range(0,13):
    index_dataset = lr_datasets.index[i]
    slopes.append(lr_datasets[index_dataset].slope)
    intercepts.append(lr_datasets[index_dataset].intercept)
    rvalues.append(lr_datasets[index_dataset].rvalue)

df_lr_datasets = pd.DataFrame(data=list(zip(slopes, intercepts, rvalues)), columns=['Slopes', 'Intercepts', 'R values'])
df_lr_datasets = df_lr_datasets.set_index(lr_datasets.index)

df_lr_datasets


# In[18]:


import seaborn as sns
grid_scatterplots = sns.FacetGrid(data, col="dataset", hue="dataset", col_wrap=4)
grid_scatterplots.map_dataframe(sns.scatterplot, x="x", y="y")


# In[19]:


sns.lmplot(data=data, x="x", y="y", col="dataset", hue="dataset", line_kws={'color': '#003f5c'}, col_wrap=4, ci=None, height=3)


# In[ ]:




