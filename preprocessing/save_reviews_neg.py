# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:17:26 2016

@author: silvia
"""

import os
import glob

list_of_files = glob.glob('./neg/*.txt')
i = 0
for fileName in list_of_files:
    i = i + 1
    print(i)
    fin = open( fileName, "r" )
    data_list = fin.readlines()
    fin.close() # closes file
    
    fout = open("neg.txt", "a")
    fout.writelines(data_list)
    fout.writelines("\n")
    fout.close()
    
   
