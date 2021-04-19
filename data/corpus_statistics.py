"""
Script to calculate word frequencies and other statistics from a file 'sentences.txt'
containing one sentence per line.
Writes the files:
 - words.txt: with each line containing 'word frequency', sorted by frequency
 - statistics: some statistics on the corpus of sentences
 - sentence_length.pdf: plot of the sentence length distribution
 - rare_words.pdf: plot of the distribution of rare words
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

corpus_path = 'sentences.txt'


def count_words(s: str) -> int:
    count = 1
    for c in s:
        if c == ' ':
            count += 1
    return count


def collect_stats(sentences):
    word_freq = defaultdict(lambda : 0)
    commas = []
    lengths = []
    lengths_with_comma = []
    
    for s in sentences:
        words = s[:-2].replace(',','').split(' ')       
        for word in words:
            word_freq[word] += 1
        l = count_words(s)
        lengths.append(l)
        c = s.count(',')
        commas.append(c)
        if c:
            lengths_with_comma.append(l)

    word_freq.default_factory = None # from now on raise key error for unknown key
 
    return word_freq, commas, lengths, lengths_with_comma


def percentage_rare(word_freq, threshold=1):
    n = 0
    for freq in word_freq.values():
        if freq <= threshold:
            n += 1            
    return n / len(word_freq) * 100 


def num_rare_words(sentence, word_freq, threshold=1):
    count = 0
    words = sentence[:-2].replace(',','').split(' ') 
    for word in words:
        if word_freq[word] <= threshold:
            count += 1
    return count


print('Reading file:', corpus_path)
with open(corpus_path, mode='r') as f:
    sentences = f.readlines()
    print('Collecting statistics')
    word_freq, commas, lengths, lengths_with_comma = collect_stats(sentences)

    
rare_freq = {x: percentage_rare(word_freq, x) for x in [1,5,10,25,50]}
comma_percent = int((len(sentences) - commas.count(0)) / len(sentences) * 100 )

stats = {
    'No of words in the corpus': sum(word_freq.values()),
    'No of different words in the corpus': len(word_freq),
    'Percentage of words that just occur once': rare_freq[1],
    'Percentage of words that occur up to five times': rare_freq[5],
    'Percentage of words that occur up to ten times': rare_freq[10],
    'Percentage of words that occur up to 25 times': rare_freq[25],
    'Percentage of words that occur up to 50 times': rare_freq[50],
    'No of sentences': len(sentences),
    'Percentage of sentences with one or more commas': comma_percent}
    

with open('statistics.txt', mode='w') as f:
    for key, val in stats.items():
        f.write(key + ': ' + '{0:,}'.format(val) + '\n')


def plot_sentence_length(len_all, len_comma):

    fig, ax = plt.subplots()

    ax.set_title('Length of sentences in corpus') 
    ax.set_xlabel('Length in words')
    ax.set_ylabel('Number of sentences')

    ax.hist(len_all,   bins=40, range=(0,40), alpha=0.5, label='all')
    ax.hist(len_comma, bins=40, range=(0,40), alpha=0.5, label='with comma')
    ax.legend() # default is 'upper right'

    fig.subplots_adjust(left=0.15, right=0.95)
    fig.savefig('sentence_length.png')


print('Plotting sentence lengths')
plot_sentence_length(lengths, lengths_with_comma)


def plot_rare_words(sentences, word_freq):
    """ Plot how many rare words the sentences contain. Rare means,
    occuring less than k times in the corpus"""
    
    fig = plt.figure(figsize=(11,3), constrained_layout=True)
    axs = fig.subplots(1,5)
    cases = [1,5,10,25,50]
    n = 5
    
    for ax, case in zip(axs, cases):
        x = [num_rare_words(s, word_freq, threshold = case) for s in sentences]
        bins, counts = np.unique(x, return_counts=True)

        ax.set_aspect(5/100)
        ax.set_ylim([0,100])
        ax.set_xticks(range(5))
        ax.set_title('k=%s' % str(case))
        
        ax.bar(bins[:n], counts[:n]/len(x)*100)

    fig.suptitle('Percentage of sentences with rare words')
    fig.text(.33,.87, 'rare means, occuring less or equal than k times in the corpus')
    
    axs[0].set_ylabel('sentences (in %)')
    axs[2].set_xlabel('rare words')
    
    fig.savefig('rare_words.png')

print('Plotting rare word frequency')
plot_rare_words(sentences, word_freq)


snd = lambda x : x[1]
words_desc = list(word_freq.items())
words_desc.sort(key=snd, reverse=True)


print('Writing word frequencies')
with open('words.txt', mode='w') as f:
    for word, freq in words_desc:
        f.write(word + ' ' + str(freq) + '\n')
