


% Preprocessing of book reviews for CNN implementation
book_Data = load('book_DataSet.txt');
vocabulary = textread('book_vocabList.txt', '%s');

vocabulary1 = cell(length(vocabulary), 5);

for i = 1: length(vocabulary)
    disp(i)
    a = vocabulary{i,1};
    num_split = strsplit(a,'_');
    for j = 1: length(num_split)
        vocabulary1(i,j) = num_split(j);  
    end
    %vocabulary(i,2) = {str2num(vocabulary{i,2})};
end

stoplist = textread('stoplist.txt', '%s');
%add empty strings
for i = 1:length(vocabulary1)
    for j = 1:2
        if isempty(vocabulary1{i,j})
            vocabulary1{i, j} = ' ';
        end
    end
end

stoplist = unique(stoplist);
p = 0;
for i =1: size(stoplist, 1);
    a = stoplist(i);
    word = a(1);
    colm1 = vocabulary1(:,1);
    index = strmatch(word, colm1, 'exact');
    for k = 1:length(index)
        disp(k)
        vocabulary1{index(k),1} = ' ';
    end
    
    colm2 = vocabulary1(:,2);
    index = strmatch(word, colm2, 'exact');
    for k = 1:length(index)
        vocabulary1{index(k),2} = ' ';
    end
end


for i = 1:length(vocabulary1)
    for j = 1:2
        if vocabulary1{i,j} == ' ' 
            vocabulary1{i,j} = [];
        end
    end
end

for i = 1:length(vocabulary1)
    disp(i);
%     vocabulary_new = [vocabulary1{i,1},' ', vocabulary1{i,2}];
%     vocab(i, :) = vocabulary_new;
    vocabulary_new = [vocabulary1{i,1},' ', vocabulary1{i,2}];
    vocal(i,1) = {vocabulary_new};
end

vocabulary = vocal;
book_DataSet = [book_Data(:,1), (book_Data(:,2)+1), book_Data(:,3)];

label = load('book_Label.txt');
stoplist = textread('stoplist_reviews.txt', '%s');

% find the index for the stoplist words in vocabulary
stoplist = unique(stoplist);
p = 0;
remove_list = [];
for i =1: size(stoplist, 1);
    a = stoplist(i);
    word = a(1);
    index = strmatch(word, vocabulary, 'exact');
    if ~isempty(index)
        p = p + 1;
        %remove_list(p,1) = index;
        remove_list = [remove_list; index];
    end
end

% remove words from book dataset
book_index1 = [];
for i = 1:size(remove_list,1)
    book_index = find(book_DataSet(:,2) == remove_list(i)); 
    book_index1 = [book_index1; book_index];
end
book_data_ind = 1:length(book_DataSet);
book_index_diff = setdiff(book_data_ind, book_index1);
book_DataSet = book_DataSet(book_index_diff,:);

% remove words from vocabulary
vocab_ind = 1:length(vocabulary);
vocab_diff = setdiff(vocab_ind, remove_list);
vocabulary_new = vocabulary(vocab_diff,:);

for i = 1: 2000
    index = find(book_DataSet(:,1) == i);
    doc_label = label(i);

    if doc_label == 1
        if length(index) <= 650
            cd('/Users/silvia/Documents/Courses/2016/503/Project/CNN-for-Sentence-Classification-in-Keras-master/book_reviews/acl_dataset/documents/pos');

            filename = ['document' num2str(i) '.txt'];
            fileID = fopen(filename,'a');
            length_pos(i) = length(index); 
            for j = 1:length(index)
                %for k = 1:book_DataSet(index(j),3)
                fprintf(fileID,vocabulary{book_DataSet(index(j),2)});
                fprintf(fileID,' ');
                %end     
            end
            fclose(fileID);
        end
    else
        if length(index) <= 650
            length_neg(i) = length(index); 
            cd('/Users/silvia/Documents/Courses/2016/503/Project/CNN-for-Sentence-Classification-in-Keras-master/book_reviews/acl_dataset/documents/neg'); 
            filename = ['document' num2str(i) '.txt'];
            fileID = fopen(filename,'a');
        
            for j = 1:length(index)
                %for k = 1:book_DataSet(index(j),3)
                fprintf(fileID,vocabulary{book_DataSet(index(j),2)});
                fprintf(fileID,' ');
                %end

            end
            fclose(fileID);   
        end
    end
%     filename = ['document' num2str(i) '.txt'];
%     fileID = fopen(filename,'a');
%     for j = 1:length(index)
%         for k = 1:book_DataSet(index(j),3)
%             fprintf(fileID,book_vocabularyList{(book_DataSet(index(j),2)+1)});
%             fprintf(fileID,' ');
%         end
%         
%     end
%     fclose(fileID);
%     
    
end