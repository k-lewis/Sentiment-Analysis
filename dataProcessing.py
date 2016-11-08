
# coding: utf-8

# In[1]:

import numpy as np
import re


# In[11]:

fpos = open("/Users/marcia/Documents/ec503/processed_acl/books/positive.review", 'r')
fneg = open("/Users/marcia/Documents/ec503/processed_acl/books/negative.review", 'r')


# In[12]:

docNo=1
Doclist=[]
wordlist=[]
numlist=[]
uniqueWordList=[]
label=[]
for line in fpos:
    words = re.split(':| ',line)
    wordlen = int(len(words))-2;
    for i in np.arange(0,wordlen,2):
        Doclist.append(docNo)
        wordlist.append(words[i])
        numlist.append(words[i+1])
        if words[i] in uniqueWordList: continue
        uniqueWordList.append(words[i])
    label.append(1)
    docNo=docNo+1
for line in fneg:
    words = re.split(':| ',line)
    wordlen = int(len(words))-2;
    for i in np.arange(0,wordlen,2):
        Doclist.append(docNo)
        wordlist.append(words[i])
        numlist.append(words[i+1])
        if words[i] in uniqueWordList: continue
        uniqueWordList.append(words[i])
    label.append(2)
    docNo=docNo+1
fpos.close()
fneg.close()


# In[13]:

flabel = open('book_Label.txt', 'w')
for item in label:
  flabel.write("%s\n" % item)
flabel.close()


# In[5]:

wordIDlist=[]
for word in wordlist:
    wordid = uniqueWordList.index(word)
    wordIDlist.append(wordid)


# In[6]:

fvocab = open('book_vocabList.txt', 'w')
for item in uniqueWordList:
  fvocab.write("%s\n" % item)
fvocab.close()


# In[7]:

DataSet= list(zip(Doclist, wordIDlist, numlist))


# In[8]:

f = open('book_DataSet.txt', 'w')
for item in DataSet:
  f.write("%s %s %s\n" % item)
f.close()


# In[9]:

len(uniqueWordList)


# In[10]:

len(DataSet)


# In[ ]:



