import string
import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import Counter

def load_data(path='data/us_trial', max_example=None):
    """
        Load data from '{path}.{text, labels}'
    """
    num_examples = 0
    tweets, emojis = [], []
    f_x, f_y = open(path + '.text', 'r'), open(path + '.labels', 'r')
    while True:
        tweet, emoji = f_x.readline(), f_y.readline()
        if not tweet or not emoji:
            break

        # TODO: extra preprocessing step for each tweet e.g. take care of slang
        tweet = tweet.strip().lower().translate({ord(c): None for c in '@#'}) # @ user -> user, #omg -> omg
        tknzr = TweetTokenizer()
        words = tknzr.tokenize(tweet)
        tweet = ' '.join(words)

        tweets.append(tweet)
        emojis.append(int(emoji)) # convert '7' -> 7

        num_examples += 1
        if (max_example is not None) and (num_examples >= max_example):
            break

    return tweets, emojis

def build_dict(tweets, max_words=50000):
    """
        Build a dictionary for words in tweets
        Only max_words are kept and the remaining will be mapped to <UNK>
    """
    wcount = Counter()
    for tweet in tweets:
        for word in tweet.split(' '):
            wcount[word] += 1

    ls = wcount.most_common(max_words)
    print("#Words: %d -> %d" %  (len(wcount), len(ls)))

    # leave 0 to UNK
    return {w[0]: index + 1 for (index, w) in enumerate(ls)}


def generate_embeddings(word_dict, dim, pretrained_path=None):
    '''
        Generate the intial embedding matrix
        If no pretrained weight is given or the word is not in the given embedding file,
        a randomly initialized weight will be used
    '''
    num_words = max(word_dict.values()) + 1 # <UNK>
    embeddings = np.random.uniform(low=-0.01, high=0.01, size=(num_words, dim))
    print('Embedding Matrix: %d x %d' % (num_words, dim))

    if pretrained_path is not None:
        pre_trained = 0
        for line in open(pretrained_path).readlines():
            v = line.split()
            if v[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[v[0]]:] = [float(x) for x in v[1:]]
        print('Pre-trained: %d (%.2f%%)' %
                        (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings
