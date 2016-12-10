load feature_matrix_neg.mat
load feature_matrix_pos.mat

% find number of samples in each class
n_pos = size(feature_matrix_pos,1);
n_neg = size(feature_matrix_neg,1);
% create label vector with pos_class = 1, neg_class = 0
labels = zeros(n_neg + n_pos, 1);
labels(1:n_pos) = 1;
% split up samples: 90% training, 10% test
data = vertcat(feature_matrix_pos, feature_matrix_neg);
% generate 1106 random indices for test samples
s = RandStream('mt19937ar','Seed',0);
indices = randperm(s, n_neg + n_pos);
test_samples = data(indices(1:1106),:);
test_labels = labels(indices(1:1106));
training_samples = data(indices(1107:length(indices)),:);
training_labels = labels(indices(1107:length(indices)),:);

% separate training data into 5 cross validation groups
% 4 groups: 1991, 1 group: 1992
s = RandStream('mt19937ar','Seed',1);
indices = randperm(s,length(training_labels));
cv_1 = sparse(training_samples(indices(1:995),:));
labels_1 = training_labels(indices(1:995),:);

cv_2 = sparse(training_samples(indices(996:1990),:));
labels_2 = training_labels(indices(996:1990),:);

cv_3 = sparse(training_samples(indices(1991:2985),:));
labels_3 = training_labels(indices(1991:2985),:);

cv_4 = sparse(training_samples(indices(2986:3980),:));
labels_4 = training_labels(indices(2986:3980),:);

cv_5 = sparse(training_samples(indices(3981:4976),:));
labels_5 = training_labels(indices(3981:4976),:);

cv_6 = sparse(training_samples(indices(4977:5972),:));
labels_6 = training_labels(indices(4977:5972),:);

cv_7 = sparse(training_samples(indices(5973:6968),:));
labels_7 = training_labels(indices(5973:6968),:);

cv_8 = sparse(training_samples(indices(6969:7964),:));
labels_8 = training_labels(indices(6969:7964),:);

cv_9 = sparse(training_samples(indices(7965:8960),:));
labels_9 = training_labels(indices(7965:8960),:);

cv_10 = sparse(training_samples(indices(8961:9956),:));
labels_10 = training_labels(indices(8961:9956),:);

% use cross validation to find boxconstraint value with best CCR
C_values = zeros(12,1);
C = 2^(-5);
for n=1:12
    C_values(n) = C;
    C = C * 2;
end

avg_CCRs = zeros(12,1);
CCRs = zeros(10,1);
options = statset('MaxIter',40000);

for n= 1:12
    % leave out 10
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_4, cv_5, cv_6, cv_7, cv_8, cv_9), vertcat(labels_1,labels_2,labels_3,labels_4, labels_5,labels_6, labels_7, labels_8, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_10);
    CCRs(1) = sum(svm_predict==labels_10)/length(svm_predict);
    clear svm_train svm_predict 
    
    % leave out 9
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_4, cv_5, cv_6, cv_7, cv_8, cv_10), vertcat(labels_1,labels_2,labels_3,labels_4, labels_5,labels_6, labels_7, labels_8, labels_10), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_9);
    CCRs(2) = sum(svm_predict==labels_9)/length(svm_predict);
    clear svm_train svm_predict 
    
    % leave out 8
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_4, cv_5, cv_6, cv_7, cv_10, cv_9), vertcat(labels_1,labels_2,labels_3,labels_4, labels_5,labels_6, labels_7, labels_10, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_8);
    CCRs(3) = sum(svm_predict==labels_8)/length(svm_predict);
    clear svm_train svm_predict
    
    % leave out 7
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_4, cv_5, cv_6, cv_10, cv_8, cv_9), vertcat(labels_1,labels_2,labels_3,labels_4, labels_5,labels_6, labels_10, labels_8, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_7);
    CCRs(4) = sum(svm_predict==labels_7)/length(svm_predict);
    clear svm_train svm_predict 
    
    % leave out 6
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_4, cv_5, cv_10, cv_7, cv_8, cv_9), vertcat(labels_1,labels_2,labels_3,labels_4, labels_5,labels_10, labels_7, labels_8, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_6);
    CCRs(5) = sum(svm_predict==labels_6)/length(svm_predict);
    clear svm_train svm_predict
   
    % leave out 5
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_4, cv_10, cv_6, cv_7, cv_8, cv_9), vertcat(labels_1,labels_2,labels_3,labels_4, labels_10,labels_6, labels_7, labels_8, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_5);
    CCRs(6) = sum(svm_predict==labels_5)/length(svm_predict);
    clear svm_train svm_predict 
    
    % leave out 4
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_10, cv_5, cv_6, cv_7, cv_8, cv_9), vertcat(labels_1,labels_2,labels_3,labels_10, labels_5,labels_6, labels_7, labels_8, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_4);
    CCRs(7) = sum(svm_predict==labels_4)/length(svm_predict);
    clear svm_train svm_predict 
    
    % leave out 3
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_10,cv_4, cv_5, cv_6, cv_7, cv_8, cv_9), vertcat(labels_1,labels_2,labels_10,labels_4, labels_5,labels_6, labels_7, labels_8, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_3);
    CCRs(8) = sum(svm_predict==labels_3)/length(svm_predict);
    clear svm_train svm_predict
    
    % leave out 2
    svm_train = svmtrain(vertcat(cv_1,cv_10,cv_3,cv_4, cv_5, cv_6, cv_7, cv_8, cv_9), vertcat(labels_1,labels_10,labels_3,labels_4, labels_5,labels_6, labels_7, labels_8, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_2);
    CCRs(9) = sum(svm_predict==labels_2)/length(svm_predict);
    clear svm_train svm_predict 
    
    % leave out 1
    svm_train = svmtrain(vertcat(cv_10,cv_2,cv_3,cv_4, cv_5, cv_6, cv_7, cv_8, cv_9), vertcat(labels_10,labels_2,labels_3,labels_4, labels_5,labels_6, labels_7, labels_8, labels_9), 'kernel_function', 'linear', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options);
    svm_predict = svmclassify(svm_train,cv_1);
    CCRs(10) = sum(svm_predict==labels_1)/length(svm_predict);
    clear svm_train svm_predict
   
    % average over folds 
    avg_CCRs(n) = sum(CCRs)/10;
end


% find best C value
[~,index] = max(avg_CCRs);
best_C = C_values(index);

% plot CCR vs C values
figure
plot(log2(C_values), avg_CCRs);
xlabel('log2(C Values)');
ylabel('CCR Values');
title('CCR vs C Values');

% use best C to test dataset
svm_train = svmtrain(vertcat(cv_1, cv_2, cv_3, cv_4, cv_5, cv_6, cv_7,cv_8, cv_9, cv_10), vertcat(labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_7, labels_8, labels_9, labels_10), 'kernel_function','linear','options',options, 'autoscale','false','boxconstraint',best_C,'kernelcachelimit',500000);
svm_classify = svmclassify(svm_train, test_samples);
test_CCR = sum(svm_classify == test_labels)/length(svm_classify);
confusion_matrix = confusionmat(test_labels, svm_classify);
