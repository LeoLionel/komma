"""
Script to compile the traing data from two input files:
 - sentences.txt: containing one sentence per line, created by 'build_corpus.py'
 - words.txt: one word and its frequency per line (sorted), created by 'corpus_statistics.py'

in addition the file:
 'cc.de.300.bin'
of precalculated german fasttext word vectors is needed. Avaliable here:
https://fasttext.cc/docs/en/crawl-vectors.html#models
Download link:
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz

The script writes the training data into two files
  dataset-v2-wv_dicts.pickle: containing wordvectors, index to word and word to index 
                              dictionaries
  dataset-v2-xy_pairs.npy: containing the input sequences (x) and their labels (y)
"""

from collections import defaultdict
import numpy as np
import pickle
import random

import fasttext # for pretrained word embeddings

path_sentences = 'sentences.txt'
path_words = 'words.txt'
path_fasttext = 'cc.de.300.bin'


max_sentence_len = 35
rare_word_threshold = 50 
max_unk_per_sentence = 2
min_num_commas = 0
save_name = 'dataset-v2-all-ml35-4460k'


special_tokens = {
    '<pad>':   0,
    '<eos>':   1,  # end of sentence
    '<comma>': 2,
    '<unk>':   3   # unknown word
}  


return_unk = lambda: special_tokens['<unk>']
word_to_ix = defaultdict(return_unk)
ix_to_word = {}


print('Building dictionaries...')
for token, ix in special_tokens.items():
    word_to_ix[token] = ix
    ix_to_word[ix] = token

    
with open(path_words, mode='r') as f:
    lines = f.readlines()
    n = len(special_tokens)
    for ix, line in enumerate(lines, start=n):
        try:
            word, freq = line.strip().split(' ')
        except ValueError:
            print('Value error, can not split line:', line)
        if int(freq) < rare_word_threshold:
            break
        else:
            word_to_ix[word] = ix
            ix_to_word[ix] = word


def pad(xs: np.ndarray, n: int) -> np.ndarray:
    "Pad a numpy array with zeros up to length n"
    m = len(xs)
    if m < n:
        e = np.zeros(n-m, dtype=np.int32)
        return np.append(xs, e) 
    else:
        return xs

    
def to_training_pair(seq: np.ndarray, comma_ix) -> (np.ndarray, np.ndarray):
    """Takes a sentence as a sequence of indices and transforms it into a
    training pair x, y. 
    x: all comma tokens are removed from seq
    y: one of the three categories for the words in x: 
       1: followed by <eos>
       2: followed by <comma>
       3: followed by any word
    Caution: This function depends on the indices given to the special tokens
    (in the dict special_tokens)!
    """
    x = seq[:-1]
    y = seq[1:]
    mask = (x != comma_ix)
    y = np.minimum(y, 3)
    x = x[mask]
    y = y[mask]
    l = max_sentence_len
    return pad(x, l), pad(y, l) 


""" Example:
sentence = ['Alle', 'drei', 'Personen', '<comma>', 'die', 'auf', 'der', 'Lok', 
'waren', '<comma>', 'starben', 'hierbei']

seq = np.array([word_to_ix[word] for word in sentence] + [word_to_ix['<eos>']])
x, y = to_training_pair(seq, word_to_ix['<comma>'])

[ix_to_word[i] for i in x] == ['Alle', 'drei', 'Personen', 'die', 'auf', 'der', 
'Lok', 'waren', 'starben', 'hierbei']

list(y) == [3, 3, 2, 3, 3, 3, 3, 2, 3, 1]

"""


n_comma = 0
n_all = 0
xy_pairs = []


print('Processing sentences...') 
with open(path_sentences, mode='r') as f:
    lines = f.readlines()
    for line in lines:
        num_commas = line.count(',')
        if num_commas < min_num_commas:
            continue
        words = line[:-2].replace(', ', ' <comma> ').split(' ')
        if len(words) > max_sentence_len:
            continue       
        seq = [word_to_ix[w] for w in words]
        if seq.count(word_to_ix['<unk>']) > max_unk_per_sentence:
            continue
        seq = seq + [word_to_ix['<eos>']]
        seq = np.array(seq, dtype=np.int32)
        x, y = to_training_pair(seq, word_to_ix['<comma>'])        
        xy_pairs.append((x,y))
        if any(y == word_to_ix['<comma>']):
            n_comma += 1
        n_all += 1

print('Found', n_all, 'valid sentences')
random.shuffle(xy_pairs)

xy_pairs_arr = np.stack(xy_pairs[:4_660_000])


print('Loading word vectors...')
fasttext_model = fasttext.load_model(path_fasttext)
ft_dim = fasttext_model.get_dimension()


print('Processing word vectors...')
wordvecs = []
# wordvecs[word_to_ix[word]] should give the wordvector for word
n = len(special_tokens)
for ix in ix_to_word:
    if ix < n: # wordvecs for special tokens
        v = np.zeros(ft_dim).astype('float32')
        wordvecs.append(v)
    else:
        v = fasttext_model.get_word_vector(word).astype('float32')
        wordvecs.append(v)

wordvecs = np.stack(wordvecs)
wordvecs[word_to_ix['<unk>']] = np.average(wordvecs, axis = 0)
# From the special tokens, just the token <unk> gets a (non zero) word vector,
# since the input sequences x don't contain other special tokens than <unk>.


print('Writing data to disk...')
with open(save_name + '-wv_dicts.pickle', 'wb') as f:
    data = wordvecs, ix_to_word, word_to_ix
    word_to_ix.default_factory = None # pickle wont't work otherwise
    pickle.dump(data, f, protocol=4)

np.save(save_name + '-xy_pairs.npy', xy_pairs_arr)


print('Number of valid sentences: {}'.format(n_all))
print('Compiled trainig data with {} sentences.'.format(len(xy_pairs_arr)) )
print('Sentences with comma: {0:.2f}%'.format(n_comma / n_all * 100))        
print('Done!')

