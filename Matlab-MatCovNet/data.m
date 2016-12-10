load('input_data_sub.mat');
rng(56,'v5uniform');
[noOfSamples,sl,el] = size(embedded_data);
indexShuff = randperm(noOfSamples);

xShuffled = embedded_data(indexShuff,:,:);
yShuffled = label(indexShuff,:);

yData = zeros(length(yShuffled),1);
for i=1:length(yShuffled)
    if yShuffled(i,1) == 0
        yData(i) = 1;
    else
       yData(i) = 2;
    end
end
[trainInd,valInd,testInd] = dividerand(noOfSamples,0.9,0.0,0.1);
xData = reshape(xShuffled,[sl,el,1,noOfSamples]);
xTrain = xData(:,:,:,trainInd);
yTrain = yData(trainInd);
xTest = xData(:,:,:,testInd);
yTest = yData(testInd);

Indices = crossvalind('Kfold', length(yTrain), 10);

% xtrain1 = xData(:,:,:,Indices~=1);
% xtest1 = xData(:,:,:,Indices==1);
% xtrain2 = xData(:,:,:,Indices~=2);
% xtest2 = xData(:,:,:,Indices==2);
% xtrain3 = xData(:,:,:,Indices~=3);
% xtest3 = xData(:,:,:,Indices==3);
% xtrain4 = xData(:,:,:,Indices~=4);
% xtest4 = xData(:,:,:,Indices==4);
% xtrain5 = xData(:,:,:,Indices~=5);
% xtest5 = xData(:,:,:,Indices==5);
% xtrain6 = xData(:,:,:,Indices~=6);
% xtest6 = xData(:,:,:,Indices==6);
% xtrain7 = xData(:,:,:,Indices~=7);
% xtest7 = xData(:,:,:,Indices==7);
% xtrain8 = xData(:,:,:,Indices~=8);
% xtest8 = xData(:,:,:,Indices==8);
% xtrain9 = xData(:,:,:,Indices~=9);
% xtest9 = xData(:,:,:,Indices==9);
% xtrain10 = xData(:,:,:,Indices~=10);
% xtest10 = xData(:,:,:,Indices==10);
% % 
% ytrain1 = yData(Indices~=1);
% ytest1 = yData(Indices==1);
% ytrain2 = yData(Indices~=2);
% ytest2 = yData(Indices==2);
% ytrain3 = yData(Indices~=3);
% ytest3 = yData(Indices==3);
% ytrain4 = yData(Indices~=4);
% ytest4 = yData(Indices==4);
% ytrain5 = yData(Indices~=5);
% ytest5 = yData(Indices==5);
% ytrain6 = yData(Indices~=6);
% ytest6 = yData(Indices==6);
% ytrain7 = yData(Indices~=7);
% ytest7 = yData(Indices==7);
% ytrain8 = yData(Indices~=8);
% ytest8 = yData(Indices==8);
% ytrain9 = yData(Indices~=9);
% ytest9 = yData(Indices==9);
% ytrain10 = yData(Indices~=10);
% ytest10 = yData(Indices==10);


save('data_Sub.mat', 'xTrain', 'yTrain', 'xTest', 'yTest','Indices');