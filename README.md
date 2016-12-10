# Sentiment-Analysis

Books (Blitzer et al., 2007)

SVM

~ include how we used python to create dataset/labels/vocabulary?

Description of Code:
The books-SVM folder contains three .mat files with the vocabulary, dataset, and labels respectively. Additionally there are four .m files: 5-fold cross validation linear, 5-fold cross validation RBF, 10-fold cross validation linear, and 10-fold cross validation RBF. Each of these files loads the three .mat files, creates a frequency matrix, uses cross validation to learn the best parameter(s) and test the dataset. In this dataset, the samples are split 20% test, 80% training. The cross-validation sets and training/testing sets are split randomly using RandStream and randperm.

Running the Code:
Save the vocabulary, dataset, and labels as separate .mat files in the same directory as code
Run SVM_linear_5fold.m, SVM_linear_10fold.m, SVM_rbf_5fold.m, or SVM_rbf_10fold.m
