#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("URL_ID.csv")
df.head()


# In[20]:


stopwords=[]

stop=pd.read_csv("StopWords/StopWords_Auditor.txt",names=['stop'])
stop['stop']=stop['stop'].str.lower()
s=stop['stop'].tolist()
stopwords.extend(s)

stop = pd.read_csv("StopWords/StopWords_Currencies.txt", names=['currency'], encoding='latin-1', on_bad_lines='skip')
stop['currency'] = stop['currency'].str.split(' ')
stop['currency'] = stop['currency'].str.get(0)
stop['currency'] = stop['currency'].str.lower()
s = stop['currency'].tolist()
stopwords.extend(s)

stop=pd.read_csv("StopWords/StopWords_DatesandNumbers.txt",names=['col'],on_bad_lines='skip')
stop['col']=stop['col'].str.split(' ')
stop['col']=stop['col'].str.get(0)
stop['col']=stop['col'].str.lower()
s=stop['col'].tolist()
stopwords.extend(s)


stop=pd.read_csv("StopWords/StopWords_Generic.txt",names=['col'])
stop['col']=stop['col'].str.lower()
s=stop['col'].tolist()
stopwords.extend(s)

stop=pd.read_csv("StopWords/StopWords_Geographic.txt",names=['col'],on_bad_lines='skip')
stop['col']=stop['col'].str.split(' ')
stop['col']=stop['col'].str.get(0)
stop['col']=stop['col'].str.lower()
s=stop['col'].tolist()
stopwords.extend(s)

stop=pd.read_csv("StopWords/StopWords_GenericLong.txt",names=['col'])
stop['col']=stop['col'].str.lower()
s=stop['col'].tolist()
stopwords.extend(s)

stop=pd.read_csv("StopWords/StopWords_Names.txt",names=['col'],on_bad_lines='skip')
stop['col']=stop['col'].str.split(' ')
stop['col']=stop['col'].str.get(0)
stop['col']=stop['col'].str.lower()
s=stop['col'].tolist()
stopwords.extend(s)


# In[21]:


print(stopwords)


# In[24]:


pos=pd.read_csv("MasterDictionary/positive-words.txt",encoding='latin-1', names=['+ive'])
positive=pos['+ive'].tolist()
neg=pd.read_csv("MasterDictionary/negative-words.txt", encoding='latin-1', names=['-ive'])
negative=neg['-ive'].tolist()
positive


# In[25]:


clean=[]
for i in df['text']:
    try:
        txt=i.split()
        result=[word for word in txt if word not in stopwords]
        txt=' '.join(result)
        clean.append(txt)
    except:
        clean.append(np.nan)
        
        


# In[26]:


df['clean']=clean


# In[27]:


import nltk
def cleaning(text):
    try:
        text=text.lower()
    except:
        return np.nan
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=" ".join(y)       
    return text


# In[28]:


df.drop(columns=['Unnamed: 0'],inplace=True)
df['clean']=df['clean'].apply(cleaning)
df.head()


# In[29]:


df['clean']=df['clean'].apply(str)
df['text']=df['text'].apply(str)


# In[30]:


def positiv(text):
    count=[]
    text=nltk.word_tokenize(text)
    for i in text:
        if i in positive:
            count.append(i)
    return len(count)
def negativ(text):
    count=[]
    text=nltk.word_tokenize(text)
    for i in text:
        if i in negative:
            count.append(i)
    return len(count)
df['POSITIVE SCORE']=df['clean'].apply(positiv)
df['NEGATIVE SCORE']=df['clean'].apply(negativ)
df['POLARITY SCORE']=(df['POSITIVE SCORE']-df['NEGATIVE SCORE'])/(df['POSITIVE SCORE']+df['NEGATIVE SCORE'])+0.000001
df.head()


# In[34]:


df['Total words C']=df['clean'].apply(lambda x: len(nltk.word_tokenize(x)))
df['SUBJECTIVITY SCORE']=(df['POSITIVE SCORE']+df['NEGATIVE SCORE'])/(df['Total words C'])+0.000001
df['Total sentences']=df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
df['Total words']=df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['AVG SENTENCE LENGTH']=df['Total words']/df['Total sentences']
df.head()


# In[41]:


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
def complex(text):
    count=0
    text=nltk.word_tokenize(text)
    for i in text:
        if syllable_count(i)>2:
            count+=1
    return count
df['Complex words']=df['text'].apply(lambda x: complex(x))
df['PERCENTAGE OF COMPLEX WORDS']=df['Complex words']/df['Total words']
 
df.head()


# In[42]:


df['FOG INDEX']=0.4*(df['AVG SENTENCE LENGTH']+df['PERCENTAGE OF COMPLEX WORDS'])
df['AVG NUMBER OF WORDS PER SENTENCE']=df['AVG SENTENCE LENGTH']


# In[43]:


def syllable(text):
    count=0
    text=nltk.word_tokenize(text)
    for i in text:
        count=count+syllable_count(i)
    return count


# In[44]:


df['Total syllable']=df['text'].apply(lambda x: syllable(x))
df['SYLLABLE PER WORD']=df['Total syllable']/df['Total words']


# In[45]:


def charac(text):
    count=0
    text=nltk.word_tokenize(text)
    for i in text:
        for j in i:
            count+=1
    return count


# In[46]:


df['Toatl char']=df['text'].apply(lambda x: charac(x))
df['AVG WORD LENGTH']=df['Toatl char']/df['Total words']
df.head()


# In[48]:


pronoun=['I','we','my','My','We','Ours','ours','Us','us']
def pronouns(text):
    count=0
    text=nltk.word_tokenize(text)
    for i in text:
        if i in pronoun:
            count+=1
    return count


df['PERSONAL PRONOUNS']=df['text'].apply(lambda x: pronouns(x))
df.head()


# In[50]:


df.columns


# In[60]:


df.rename(columns = {'Complex words':'COMPLEX WORD COUNT','Total words C':'WORD COUNT',}, inplace = True)
RESULT=df[['URL_ID','URL','POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH']]


# In[61]:


RESULT


# In[62]:


RESULT.to_excel('Output Data Structure.xlsx')


# In[ ]:




