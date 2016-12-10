% 5531 samples per class
% algorithm
    % create feature matrix n x d
    % find each word in vocab list to get id number & increment count
    
vocab = textread('vocabList.txt', '%s');
d = length(vocab);
% preallocate space
feature_matrix_pos = zeros(5531,d);
feature_matrix_neg = zeros(5531,d);

% read in positive samples & parse each line
fid = fopen('rt-polarity.pos');
tline = fgetl(fid);
counter = 1;
while ischar(tline)
    % split line into words
    words = strsplit(tline, ' ');
    word_count = 0;
    for i = 1:length(words)
	% check if word is in vocabulary and, if so, get its index
        word_id = find(ismember(vocab, words(i)));
        if(isempty(word_id)~= 1)
	  % keep track of valid words in sample (use to normalize frequency matrix)
          word_count = word_count + 1;
	  % increment count for that word id
          feature_matrix_pos(counter,word_id) = feature_matrix_pos(counter,word_id) + 1;
        end
    end
    % normalize
    feature_matrix_pos(counter, :) = feature_matrix_pos(counter, :)/word_count;
    % increment sample count
    counter = counter + 1;
    tline = fgetl(fid);
end
fclose(fid);

% read in negative samples & parse each line using same process as above
fid = fopen('rt-polarity.neg');
tline = fgetl(fid);
counter = 1;
while ischar(tline)
    words = strsplit(tline, ' ');
    word_count = 0;
    for i = 1:length(words)
        word_id = find(ismember(vocab, words(i)));
        if(isempty(word_id)~= 1)
          word_count = word_count + 1;
          feature_matrix_neg(counter,word_id) = feature_matrix_neg(counter,word_id) + 1;
        end
    end
    feature_matrix_neg(counter,:) = feature_matrix_neg(counter,:)/word_count;
    counter = counter + 1;
    tline = fgetl(fid);
end
fclose(fid);
