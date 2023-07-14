#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np


# In[41]:


xlsx_file = pd.read_excel('Input.xlsx')


# In[42]:


df['URL'][0]


# In[44]:


headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}


# In[48]:


webpage=requests.get('https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/',headers=headers).text
soup=BeautifulSoup(webpage,'lxml')
print(soup.prettify())


# In[52]:


title_element = soup.find('title')
if title_element is not None:
    title = title_element.text.strip()
else:
    title = "Title not found"
title   


# In[53]:


soup.find('div',class_="td-post-content").text


# In[55]:


content_element = soup.find('div', class_="td-post-content")
if content_element is not None:
    content = content_element.get_text(separator='\n\n').strip()
    # Remove CSS styles
    content = re.sub(r'\s*{[^}]+}\s*', '', content)
else:
    content = "Content not found"

print(content)


# In[57]:


title=[]
text=[]
for i in df['URL']:
    webpage=requests.get(i,headers=headers).text
    soup=BeautifulSoup(webpage,'lxml')
    try:
        title.append(soup.find('h1',class_="entry-title").text)
        flag=1
    except AttributeError:
        title.append(np.nan)
    try:
        text.append(soup.find('div',class_="td-post-content").text)
    except AttributeError:
        text.append(np.nan)
        
title


# In[58]:


text


# In[59]:


URL_ID=df[['URL_ID','URL']]
URL_ID.head()


# In[60]:


URL_ID['title']=title
URL_ID['text']=text
URL_ID['text']=URL_ID['text'].str.replace('\n',' ')
URL_ID.head()


# In[61]:


URL_ID.to_csv('URL_ID.csv')


# In[ ]:




