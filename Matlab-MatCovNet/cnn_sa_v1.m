% path to matconvnet
addpath('/home/marcia/matconvnet-1.0-beta23/matlab')
run vl_setupnn ;

%Batch
xbatch = xtrain(:,:,:,1:20);
ybatch = ytrain(1:20,:);


%initialize weights for Conv-Layer
rng(9,'v5normal')
W1_Conv = randn(3,128,1,155);
W2_Conv = randn(4,128,1,155);
W3_Conv = randn(5,128,1,155);

b1_Conv = zeros(1,155);
b2_Conv = zeros(1,155);
b3_Conv = zeros(1,155);

W_full = randn(465,1,1,2);
b_full = zeros(1,2);

% Convolution Layer
out_Conv1 = vl_nnconv(xbatch, W1_Conv, b1_Conv);
out_Conv2 = vl_nnconv(xbatch, W2_Conv, b2_Conv);
out_Conv3 = vl_nnconv(xbatch, W3_Conv, b3_Conv);

%ReLU layer
out_ReLu1 = vl_nnrelu(out_Conv1);
out_ReLu2 = vl_nnrelu(out_Conv2);
out_ReLu3 = vl_nnrelu(out_Conv3);

%Maxpooling layer
out_MaxPool1 = vl_nnpool(out_ReLu1,[54,1]);
out_MaxPool2 = vl_nnpool(out_ReLu2,[53,1]);
out_MaxPool3 = vl_nnpool(out_ReLu3,[52,1]);

%DropOut Layer
out_MaxPool=cat(3,out_MaxPool1,out_MaxPool2,out_MaxPool3);
out_Max = reshape(out_MaxPool,465,1,1,20);
[out_Drop, Mask] = vl_nndropout(out_Max,'rate',0.5);

%Fully Connected Layer
out_FullyCon = vl_nnconv(out_Drop,W_full,b_full);
% out_FC = reshape(out_FullyCon,1,2,1,20);

%Loss
out_Soft = vl_nnsoftmax(out_FullyCon);
out_loss = vl_nnloss(out_Soft, ybatch);

%Backpropagation
d_out_loss = vl_nnloss(out_Soft, ybatch,out_loss);

d_out_Soft = vl_nnsoftmax(out_FullyCon,d_out_loss);

d_out_FullyCon = vl_nnconv(out_Drop,W_full,b_full,d_out_Soft);

d_out_Drop = vl_nndropout(out_Max,d_out_FullyCon,'mask',Mask);

d_out_RE_Drop = reshape(d_out_Drop,1,1,465,20);

d_out_MaxPool1 = vl_nnpool(out_ReLu1,[54,1],d_out_RE_Drop(:,:,1:155,:));
d_out_MaxPool2 = vl_nnpool(out_ReLu2,[53,1],d_out_RE_Drop(:,:,156:310,:));
d_out_MaxPool3 = vl_nnpool(out_ReLu3,[52,1],d_out_RE_Drop(:,:,311:465,:));

d_out_ReLu1 = vl_nnrelu(out_Conv1,d_out_MaxPool1);
d_out_ReLu2 = vl_nnrelu(out_Conv2,d_out_MaxPool2);
d_out_ReLu3 = vl_nnrelu(out_Conv3,d_out_MaxPool3);

d_out_Conv1 = vl_nnconv(xbatch, W1_Conv, b1_Conv,d_out_ReLu1);
d_out_Conv2 = vl_nnconv(xbatch, W2_Conv, b2_Conv,d_out_ReLu2);
d_out_Conv3 = vl_nnconv(xbatch, W3_Conv, b3_Conv,d_out_ReLu3);

