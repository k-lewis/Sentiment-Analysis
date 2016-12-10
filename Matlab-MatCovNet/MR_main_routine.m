function [net, info] = MR_main_routine(varargin)
%CNN_Movie Review: Demonstrates training 

% path to matconvnet
addpath('/home/marcia/matconvnet-1.0-beta23/matlab')
run vl_setupnn;

% Path to data and stored network
root = fileparts(fileparts(mfilename('fullpath')));
opts.dataDir = fullfile(root, 'data') ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataPath = fullfile(opts.dataDir,'data_MR.mat');
opts.expDir = fullfile(root, 'data', 'processedData') ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
numEpochs = 500;
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
% 10 fold cross-validation
for k=1:10
    opts.cvPath = fullfile(opts.expDir,char(num2str(k)));
    net = movie_CNN(); % initialize the network
    net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:2,'UniformOutput',false);
    net.meta.trainOpts.numEpochs = numEpochs ;
    imdb = getImdDB(opts,k);
    fbatch = @(i,b) getBatch(opts.train,i,b);
    [net, info] = cnn_train_dag(net, imdb, fbatch, ...
                                'expDir', opts.cvPath, ...
                                net.meta.trainOpts, ...
                                opts.train, ...
                                'val', find(imdb.images.set == 2)) ;
end

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------

if ~isa(imdb.images.data, 'gpuArray') && numel(opts.gpus) > 0
  imdb.images.data = gpuArray(imdb.images.data);
  imdb.images.labels = gpuArray(imdb.images.labels);
end
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getImdDB(opts,k)
% --------------------------------------------------------------------

% Prepare the IMDB structure:
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end
if ~exist(opts.dataPath)
  fprintf('Downloading %s to %s.\n', opts.dataURL, opts.dataPath) ;
  urlwrite(opts.dataURL, opts.dataPath) ;
end
dat = load(opts.dataPath);
data = single(dat.xTrain);
set = [ones(1,numel(dat.yTrain(dat.Indices~=k))) 2*ones(1,numel(dat.yTrain(dat.Indices==k)))];
train = data(:,:,:,dat.Indices~=k);
test = data(:,:,:,dat.Indices==k);
imdb.images.data = cat(4,train,test);
imdb.images.labels = single(cat(1, dat.yTrain(dat.Indices~=k),dat.yTrain(dat.Indices==k)))' ;
imdb.images.set = set ;
imdb.meta.sets = {'train','test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:2,'uniformoutput',false) ;