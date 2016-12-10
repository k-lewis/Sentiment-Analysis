clear; close all; clc;

embedding = textread('embedding.txt', '%s');

for j = 1:length(embedding)
    embedding_vec(j) = str2num(embedding{j});
end
% vocabulary size 19389
% embeding size 128
embedding = reshape(embedding_vec, [128, 19389]);

fid = fopen('vocab.csv');
out = textscan(fid,'%s');
fclose(fid);

% vocabulary length = 18765
vocab_out = out{1,1};
for i = 1: length(vocab_out)
    disp(i)
    a = vocab_out{i,1};
    vocabulary(i,:) = strsplit(a,',');  
    vocabulary(i,2) = {str2num(vocabulary{i,2})};
end


% N = 5;
% fileID = fopen('data.txt');
% formatSpec = '%s';
% 
% k = 0;
% while ~feof(fileID)
% 	k = k+1;
% 	C = textscan(fileID,formatSpec,N);
% 
% end

dataset = textread('data.txt', '%s');

for i = 1:length(dataset)
    disp(i)
    dataset1(i) = str2num(dataset{i});
end
% number of samples(pos and neg) = 1898
% sequence length = 685
dataset = dataset';
data = reshape(dataset1, [685, 1898]);

label = textread('label.txt', '%s');
for i = 1:length(label)
    label1(i) = str2num(label{i,1});
end
label = reshape(label1, [2, 1898]);
label = label';

data = data';
embedding = embedding';
% place embedding in the data matrix: num_samples x seq_length x 128
for i = 1: size(data,1)
    disp(i)
    for j = 1:size(data,2)
        if data(i,j) == 0
            embedded_data(i,j,:) = zeros(1,128); 
        else
            embedded_data(i,j,:) = embedding(data(i,j),:); 
        end
    end
end

save('input_data.mat', 'data', 'label', 'vocabulary', 'embedding', 'embedded_data');

