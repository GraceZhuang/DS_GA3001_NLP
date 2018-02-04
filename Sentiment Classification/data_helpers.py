import numpy as np
import re
import itertools
from collections import Counter
import glob
import collections


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(pos_path, neg_path):
    '''
    Loan and clean texts
    '''
    files_pos=glob.glob(pos_path)
    files_neg=glob.glob(neg_path)
    data_pos=[]
    for i in files_pos:
        with open(i,'r') as f1:
            data_pos.append(f1.readlines())
            f1.close()
    data_neg=[]
    for i in files_neg:
        with open(i,'r') as f2:
            data_neg.append(f2.readlines())
            f2.close()
    data_pos = [clean_str(i[0]) for i in data_pos]
    data_neg = [clean_str(i[0]) for i in data_neg]
    x_text = data_pos + data_neg
    y_pos = [[0, 1] for _ in data_pos]
    y_neg = [[1, 0] for _ in data_neg]
    y = np.concatenate([y_pos,y_neg],0)

    return [x_text, y]

def prepare_multi_gram_text(texts, feature): # feature = 1 - 1 gram; 2 - bi-gram; 3 - tri-gram...
    '''
    Prepare multi-gram texts
    '''
    if feature == 1:
        return texts
    else:
        data = [zip(i.split(),i.split()[1:]) for i in texts]
        x_text = []
        idx = range(feature)
        for i in data:
            temp = []
            for j in i:
                temp.append(' '.join(j[x] for x in idx))
            x_text.append(temp)
        return x_text


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            

def create_dataset(data_sentence,most_freq, feature): # feature = 1 - 1 gram; 2 - bi-gram; 3 - tri-gram...

    data_words = [j for i in data_sentence for j in i.split()]

    counts = collections.Counter(data_words).most_common(most_freq)
    counts.append(('<oov>',-1))

    dictionary = dict()

    for word, _ in counts:
        dictionary[word] = len(dictionary)+feature+(most_freq*(feature-1))

    data = list()
    oov_count = 0

    for i in data_sentence:
        temp=[]
        if feature == 1:
            i = i.split()
        else:
            i = i 
        for word in i:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                oov_count += 1
            temp.append(index)
        data.append(temp)

    counts[-1] = ('<oov>',oov_count)
    freq_words = [counts[i][0] for i in range(len(counts))]
    dictionary['<oov>'] = 0
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, freq_words, dictionary, reverse_dictionary

