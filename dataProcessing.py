import numpy as np
import re

#load the positive and negative reviews
fpos = open("/Users/marcia/Documents/ec503/processed_acl/books/positive.review", 'r')
fneg = open("/Users/marcia/Documents/ec503/processed_acl/books/negative.review", 'r')

# finds unique words 
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


# write the labels

flabel = open('book_Label.txt', 'w')
for item in label:
  flabel.write("%s\n" % item)
flabel.close()


# Create the vocabulary

wordIDlist=[]
for word in wordlist:
    wordid = uniqueWordList.index(word)
    wordIDlist.append(wordid)
fvocab = open('book_vocabList.txt', 'w')
for item in uniqueWordList:
  fvocab.write("%s\n" % item)
fvocab.close()

#Create the bag of words format
DataSet= list(zip(Doclist, wordIDlist, numlist))
f = open('book_DataSet.txt', 'w')
for item in DataSet:
  f.write("%s %s %s\n" % item)
f.close()
