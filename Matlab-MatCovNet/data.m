
rng(56,'v5uniform');
indexShuff = randperm(10662);

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

xData = reshape(xShuffled,[56,128,1,10662]);
xtrain = xData(:,:,:,1:9596);
ytrain = yData(1:9596,:);
xtest = xData(:,:,:,9597:end);
ytest = yData(9597:end,:);