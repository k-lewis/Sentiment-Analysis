load labels.mat
load dataset.mat
load vocabulary.mat

num_documents = length(unique(dataset(:,1)));
W = length(vocabulary);
% create feature matrix
feature_matrix = zeros(num_documents,W);
% word_ids in dataset start at 0. Increment all word_ids by one so can use
% them as indices
dataset(:,2) = dataset(:,2) + 1;

for n=1:num_documents
    extract_rows = find(dataset(:,1) == n);
    document_rows = dataset(extract_rows,:);
    doc_length = sum(document_rows(:,3));
    for j = 1:size(document_rows,1)
        word_id = document_rows(j,2);
        word_count = document_rows(j,3);
        feature_matrix(n,word_id) = word_count/doc_length;
    end
end

feature_matrix = sparse(feature_matrix);
% randomly split the dataset : 20% test, 80% training
s = RandStream('mt19937ar','Seed',1);
rand_indices = randperm(s,2000);

test_indices = rand_indices(1:400);
test_set= feature_matrix(test_indices,:);
test_labels = labels(test_indices);

CV_1_indices = rand_indices(401:720);
CV_1 = feature_matrix(CV_1_indices,:);
CV_1_labels = labels(CV_1_indices);

CV_2_indices = rand_indices(721:1040);
CV_2 = feature_matrix(CV_2_indices,:);
CV_2_labels = labels(CV_2_indices);

CV_3_indices = rand_indices(1041:1360);
CV_3 = feature_matrix(CV_3_indices,:);
CV_3_labels = labels(CV_3_indices);

CV_4_indices = rand_indices(1361:1680);
CV_4 = feature_matrix(CV_4_indices,:);
CV_4_labels = labels(CV_4_indices);

CV_5_indices = rand_indices(1681:2000);
CV_5 = feature_matrix(CV_5_indices,:);
CV_5_labels = labels(CV_5_indices);

% make all matrices sparse for faster computation time and less memory
% storage
test_set = sparse(test_set);
CV_1 = sparse(CV_1);
CV_2 = sparse(CV_2);
CV_3 = sparse(CV_3);
CV_4 = sparse(CV_4);
CV_5 = sparse(CV_5);

% find best C value parameter for SVM linear
C = 2^(-5);
C_values = zeros(21,1);
for n = 1:21 
    C_values(n) = C;
    C = C*2;
end

avg_CCRs = zeros(21,1);
CCRs = zeros(5,1);
clear dataset labels vocabulary
% iterate through C range to find C*
for n = 1:21
    svm_train = svmtrain(vertcat(CV_1, CV_2, CV_3, CV_4),vertcat(CV_1_labels, CV_2_labels, CV_3_labels, CV_4_labels),'kernel_function','linear','autoscale','false', 'boxconstraint',C_values(n)*ones(1280,1), 'kernelcachelimit',500000);
    svm_classify = svmclassify(svm_train, CV_5);
    clear svm_train
    CCRs(1) = sum(svm_classify == CV_5_labels)/length(svm_classify);
    clear svm_classify
    
    svm_train = svmtrain(vertcat(CV_1, CV_2, CV_3, CV_5),vertcat(CV_1_labels, CV_2_labels, CV_3_labels, CV_5_labels),'kernel_function','linear', 'autoscale','false', 'boxconstraint',C_values(n)*ones(1280,1), 'kernelcachelimit',500000);
    svm_classify = svmclassify(svm_train, CV_4);
    clear svm_train
    CCRs(2) = sum(svm_classify == CV_4_labels)/length(svm_classify);
    clear svm_classify
    
    svm_train = svmtrain(vertcat(CV_1, CV_2, CV_5, CV_4),vertcat(CV_1_labels, CV_2_labels, CV_5_labels, CV_4_labels),'kernel_function','linear', 'autoscale','false','boxconstraint',C_values(n)*ones(1280,1), 'kernelcachelimit',500000);
    svm_classify = svmclassify(svm_train, CV_3);
    clear svm_train
    CCRs(3) = sum(svm_classify == CV_3_labels)/length(svm_classify);
    clear svm_classify
    
    svm_train = svmtrain(vertcat(CV_1, CV_5, CV_3, CV_4),vertcat(CV_1_labels, CV_5_labels, CV_3_labels, CV_4_labels),'kernel_function','linear','autoscale','false', 'boxconstraint',C_values(n)*ones(1280,1), 'kernelcachelimit',500000);
    svm_classify = svmclassify(svm_train, CV_2);
    clear svm_train
    CCRs(4) = sum(svm_classify == CV_2_labels)/length(svm_classify);
    clear svm_classify
    
    svm_train = svmtrain(vertcat(CV_5, CV_2, CV_3, CV_4),vertcat(CV_5_labels, CV_2_labels, CV_3_labels, CV_4_labels),'kernel_function','linear', 'autoscale','false','boxconstraint',C_values(n)*ones(1280,1), 'kernelcachelimit',500000);
    svm_classify = svmclassify(svm_train, CV_1);
    clear svm_train
    CCRs(5) = sum(svm_classify == CV_1_labels)/length(svm_classify);
    clear svm_classify
    % average over folds
    avg_CCRs(n) = sum(CCRs)/5;
end

% plot CCR vs C values to see if CCR stabilizes at end of C range or if
% need to test larger range of C values
figure 
plot(log2(C_values),avg_CCRs);
title('CCR Values vs C values for Binary SVM');
xlabel('log2(C values)');
ylabel('CCR Values');

% find C value with max CCR
[~,index] = max(avg_CCRs);
C_star = C_values(index);

% test with best C value to find final CCR
svm_train = svmtrain(vertcat(CV_1, CV_2, CV_3, CV_4, CV_5),vertcat(CV_1_labels,CV_2_labels, CV_3_labels, CV_4_labels, CV_5_labels),'kernel_function', 'linear', 'autoscale','false','boxconstraint',C_star*ones(1600,1),'kernelcachelimit', 500000);
svm_classify = svmclassify(svm_train,test_set);
test_CCR = sum(svm_classify == test_labels)/length(svm_classify);
confusion_matrix = confusionmat(test_labels,svm_classify);

