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
cv_1 = sparse(training_samples(indices(1:1991),:));
labels_1 = training_labels(indices(1:1991),:);
cv_2 = sparse(training_samples(indices(1992:3982),:));
labels_2 = training_labels(indices(1992:3982),:);
cv_3 = sparse(training_samples(indices(3983:5973),:));
labels_3 = training_labels(indices(3983:5973),:);
cv_4 = sparse(training_samples(indices(5974:7964),:));
labels_4 = training_labels(indices(5974:7964),:);
cv_5 = sparse(training_samples(indices(7965:9956),:));
labels_5 = training_labels(indices(7965:9956),:);


% use cross validation to find boxconstraint value with best CCR
C_values = zeros(12,1);
rbf_sigma_values = zeros(12,1);
rbf_sigma = 2^(-8);
C = 2^(-5);
for n=1:12
    C_values(n) = C;
    C = C * 2;
    rbf_sigma_values(n) = rbf_sigma;
    rbf_sigma = rbf_sigma*2;
end
avg_CCRs = zeros(12,12);
CCRs = zeros(5,1);
options = statset('MaxIter',40000);

for n= 1:12
    for i = 1:12
   % leave out 5
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_4), vertcat(labels_1,labels_2,labels_3,labels_4), 'kernel_function', 'rbf', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options, 'rbf_sigma', rbf_sigma_values(i));
    svm_predict = svmclassify(svm_train,cv_5);
    CCRs(1) = sum(svm_predict==labels_5)/length(svm_predict);
    clear svm_train svm_predict
    
  % leave out 4
    svm_train = svmtrain(vertcat(cv_1,cv_2,cv_3,cv_5), vertcat(labels_1,labels_2,labels_3,labels_5), 'kernel_function', 'rbf', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options, 'rbf_sigma', rbf_sigma_values(i));
    svm_predict = svmclassify(svm_train,cv_4);
    CCRs(2) = sum(svm_predict==labels_4)/length(svm_predict);
    clear svm_train svm_predict
    
   % leave out 3
    training_set = [cv_1;cv_2;cv_4;cv_5];
    training_labels = [labels_1;labels_2;labels_4;labels_5];
    svm_train = svmtrain(training_set, training_labels, 'kernel_function', 'rbf', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options, 'rbf_sigma', rbf_sigma_values(i));
    svm_predict = svmclassify(svm_train,cv_3);
    CCRs(3) = sum(svm_predict==labels_3)/length(svm_predict);
    clear svm_train svm_predict
    
  %  leave out 2;
    training_set = [cv_1;cv_4;cv_3;cv_5];
    training_labels = [labels_1;labels_4;labels_3;labels_5];
    svm_train = svmtrain(training_set, training_labels, 'kernel_function', 'rbf', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options, 'rbf_sigma', rbf_sigma_values(i));
    svm_predict = svmclassify(svm_train,cv_2);
    CCRs(4) = sum(svm_predict==labels_2)/length(svm_predict);
    clear svm_train svm_predict
    
   % leave out 1
    training_set = [cv_4;cv_2;cv_3;cv_5];
    training_labels = [labels_4;labels_2;labels_3;labels_5];
    svm_train = svmtrain(training_set, training_labels, 'kernel_function', 'rbf', 'boxconstraint',C_values(n), 'kernelcachelimit', 500000, 'autoscale', 'false', 'options', options, 'rbf_sigma', rbf_sigma_values(i));
    svm_predict = svmclassify(svm_train,cv_1);
    CCRs(5) = sum(svm_predict==labels_1)/length(svm_predict);
    clear svm_train svm_predict
   % find average of CV classifiers 
    avg_CCRs(n,i) = sum(CCRs)/5;
    end
    
end


% find best pairing of C and sigma
[~,max_CCR_index] = max(avg_CCRs(:));
[C_star_index,rbf_sigma_star_index] = ind2sub(size(avg_CCRs),max_CCR_index);
best_C = C_values(C_star_index);
rbf_sigma_best = rbf_sigma_values(rbf_sigma_star_index);

% plot sigma vs C value with CCR contour plot
figure
contourf(log2(C_values),log2(rbf_sigma_values),avg_CCRs');
colorbar
xlabel('log2(boxconstraint values)');
ylabel('log2(rbf-sigma values)');
title('Boxconstraint and rbf-sigma pair CCRs');


% use best C to test dataset
svm_train = svmtrain(vertcat(cv_1, cv_2, cv_3, cv_4, cv_5), vertcat(labels_1, labels_2, labels_3, labels_4, labels_5), 'kernel_function','rbf','rbf_sigma', rbf_sigma_best,'options',options, 'autoscale','false','boxconstraint',best_C,'kernelcachelimit',500000);
svm_classify = svmclassify(svm_train, test_samples);
test_CCR = sum(svm_classify == test_labels)/length(svm_classify);
confusion_matrix = confusionmat(test_labels, svm_classify);
