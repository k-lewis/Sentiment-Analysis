% setup MatConvNet
addpath('/home/marcia/matconvnet-1.0-beta23/matlab')
run  vl_setupnn

% load the pre-trained CNN
load('/home/marcia/EC503_CNN-SA/Sentiment-Analysis/data/processedData/5/net-epoch-100.mat')
net = dagnn.DagNN.loadobj(net) ;
net.mode = 'test' ;

% load and preprocess an image
load('/home/marcia/EC503_CNN-SA/Sentiment-Analysis/data/data_book.mat')
[~,~,~,noOfTestData] = size(xTest);
im = single(xTest) ; % note: 0-255 range
% run the CNN
bestScore = zeros(noOfTestData,1);
best= zeros(noOfTestData,1);
for i = 1:noOfTestData
    net.eval({'input', im(:,:,:,i)});
    % obtain the CNN otuput
    scores = net.vars(net.getVarIndex('x_fc_relu_out')).value ;
    scores = squeeze(gather(scores)) ;
    % show the classification results
    [bestScore(i), best(i)] = max(scores);
end
C = confusionmat(yTest,best);
ccr = sum(diag(C))/noOfTestData;