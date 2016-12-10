function nn = movie_CNN()
% hyperparameters
filter_size =[3,4,5];
num_filters = 100;
mp_size = 2;
batch_size = 50;
sequence_length = 56;
embedding_size = 128;
fc_wH = sum(fix(((((sequence_length - filter_size)+1)-mp_size)/mp_size)+1));
fc_wW = 1;
fc_wC = num_filters;
fc_num_filters = 2;


%DagNN Object
nn = dagnn.DagNN();

%dropOut layer
nn.addLayer('dropout_in', dagnn.DropOut('rate',0.25), {'input'},{'x_in_dropout'});

%Convolution Layer
c1 = dagnn.Conv('size', [filter_size(1) embedding_size 1 num_filters],...
    'pad',0,'stride',1, 'hasBias', true);
nn.addLayer('c1', c1, {'x_in_dropout'}, {'x_c1_out'}, {'cw1', 'cb1'});
c2 = dagnn.Conv('size', [filter_size(2) embedding_size 1 num_filters],...
    'pad',0,'stride',1, 'hasBias', true);
nn.addLayer('c2', c2, {'input'}, {'x_c2_out'}, {'cw2', 'cb2'});
c3 = dagnn.Conv('size', [filter_size(3) embedding_size 1 num_filters],...
    'pad',0,'stride',1, 'hasBias', true);
nn.addLayer('c3', c3, {'input'}, {'x_c3_out'}, {'cw3', 'cb3'});

%Relu layer
nn.addLayer('c_relu1', dagnn.ReLU, {'x_c1_out'}, {'x_c_relu1_out'});
nn.addLayer('c_relu2', dagnn.ReLU, {'x_c2_out'}, {'x_c_relu2_out'});
nn.addLayer('c_relu3', dagnn.ReLU, {'x_c3_out'}, {'x_c_relu3_out'});

%Maxpooling layer
c_mp1 = dagnn.Pooling('method', 'max', 'poolSize', [mp_size 1],'stride', mp_size);
nn.addLayer('c_mp1', c_mp1, {'x_c_relu1_out'}, {'x_c_mp1_out'});
c_mp2 = dagnn.Pooling('method', 'max', 'poolSize', [mp_size 1], 'stride', mp_size);
nn.addLayer('c_mp2', c_mp2, {'x_c_relu2_out'}, {'x_c_mp2_out'}); 
c_mp3 = dagnn.Pooling('method', 'max', 'poolSize', [mp_size 1], 'stride', mp_size);
nn.addLayer('c_mp3', c_mp3, {'x_c_relu3_out'}, {'x_c_mp3_out'}); 

%Concatenate max pooling layer output
nn.addLayer('concat_mp', dagnn.Concat('dim',1) , {'x_c_mp1_out','x_c_mp2_out','x_c_mp3_out'}, {'x_concat_out'});

%Dropout layer
nn.addLayer('dropout', dagnn.DropOut('rate',0.5), {'x_concat_out'},{'x_dropOut'});

%Fully Connected Layer
fc = dagnn.Conv('size',[fc_wH, fc_wW, fc_wC, fc_num_filters],'pad',0,'stride',1,'hasBias',true);
nn.addLayer('fc', fc, {'x_dropOut'}, {'x_fc_out'}, {'fcw', 'fcb'});
nn.addLayer('fc_soft', dagnn.SoftMax, {'x_fc_out'}, {'x_fc_relu_out'});

%SoftMax layer
nn.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'x_fc_relu_out','label'}, 'objective');
nn.addLayer('error', dagnn.Loss('loss', 'classerror'), {'x_fc_relu_out','label'}, 'error');

%Initial parameter
nn.initParams() ;
nn.meta.trainOpts.learningRate = 0.0001;
nn.meta.trainOpts.batchSize = batch_size;
nn.meta.trainOpts.weightDecay = 0.0005 ;
nn.meta.trainOpts.momentum = 0.9 ;
end
