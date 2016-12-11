# Sentiment-Analysis

Books (Blitzer et al., 2007)

SVM
Run dataProcessing.py under preprocessing folder to generate the dataset in bag of words format.

Description of Code:
The books-SVM folder contains three .mat files with the vocabulary, dataset, and labels respectively. Additionally there are four .m files: 5-fold cross validation linear, 5-fold cross validation RBF, 10-fold cross validation linear, and 10-fold cross validation RBF. Each of these files loads the three .mat files, creates a frequency matrix, uses cross validation to learn the best parameter(s) and test the dataset. In this dataset, the samples are split 20% test, 80% training. The cross-validation sets and training/testing sets are split randomly using RandStream and randperm.

Running the Code:
Save the vocabulary, dataset, and labels as separate .mat files in the same directory as code
Run SVM_linear_5fold.m, SVM_linear_10fold.m, SVM_rbf_5fold.m, or SVM_rbf_10fold.m

CNN
To run the CNN, install MatConvNet: http://www.vlfeat.org/matconvnet/quick/
Under Matlab-MatConvNet folder 
run books_main_routine.m to train the network
run cnn_test_books.m to test the network
The network architecture is under books_CNN.m


Movie Reviews (Pang et al., 2005)

SVM

Description of Code:
The rt-polarity-SVM folder contains five .m files, one .txt file, one .pos file, and one .neg file. The generate_features.m file reads in vocabList.txt, rt-polarity.pos, and rt-polarity.neg and creates feature matrices for the positive samples and the negative samples. The remaining .m files are for the following: 5-fold cross validation linear kernel SVM, 5-fold cross validation RBF kernel SVM, 10-fold cross validation linear kernel SVM, and 10-fold cross validation RBF kernel SVM.


Running the Code:
First, make sure all files are in the same directory. Run the generate_features.m file to generate the feature matrices for positive and negative samples. Save the feature_matrix_pos and feature_matrix_neg variables in the workspace as .mat files; these will be loaded by the remaining .m files. Once the .mat files have been created, any of the SVM files can be run in the same directory. 

CNN
To run the CNN, install MatConvNet: http://www.vlfeat.org/matconvnet/quick/
Under Matlab-MatConvNet folder 
run MR_main_routine.m to train the network
run cnn_test_MR.m to test the network
The network architecture is under movie_CNN.m
