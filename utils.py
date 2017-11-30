import string
import regex as re
import numpy as np
import splitter

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
        tweet = tweet.strip().lower() # delete whitespaces and lowercase
        tweet = tweet[:-1] if tweet[-1] == u'\u2026' else tweet # and more... -> more (... is a special unicode char)
        tweet = re.sub(r"#\S+", lambda match: ' '.join(splitter.split(match.group()[1:])), tweet) #artfactory -> art factory
        tweet = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", tweet, (re.MULTILINE | re.DOTALL)) # no wayyyyy -> no way
        tweet = tweet.translate({ord(c): None for c in '@#'}) # @ user -> user, #omg -> omg
        words = TweetTokenizer().tokenize(tweet)
        tweet = ' '.join(words)

        tweets.append(tweet)
        emojis.append(int(emoji)) # convert '7' -> 7

        num_examples += 1
        if (max_example is not None) and (num_examples >= max_example):
            break
    print("%d examples loaded" % num_examples)
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
    # print("#Words: %d -> %d" %  (len(wcount), len(ls)))

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

def generate_labels(emojis, count):
    '''
        Generate the matrix for the ground truth label results
        emojis is a list of correct labnels
        count is the total number of the classification results like 20 kinds of emojis
        For each row in the return value, the position k is labeled 1 if the label for this sentence is emoji k
        Other positions are zero value.
        This returned value is for calculating the error if I understand the model right.
        It may be unnecessary.
    '''

    labels = np.zeros([len(emojis), count])
    for i in range(len(emojis)):
        labels[i, emojis[i]] = 1.0
    return labels


def vectorize(tweets, emojis, word_dict):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    x = []
    y = []
    for i, (tweet, emoji) in enumerate(zip(tweets, emojis)):
        words = tweet.split(' ')
        seq = [word_dict[w] if w in word_dict else 0 for w in words]
        if len(seq) > 0:
            x.append(seq)
            onehoty = [0] * 20
            onehoty[emoji] = 1
            y.append(onehoty)

        if i % 10000 == 0:
            print('Vectorize: processed %d / %d' % (i, len(emojis)))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    sorted_index = len_argsort(x)
    x = [x[i] for i in sorted_index]
    y = [y[i] for i in sorted_index]

    return x, y


def pad_data(seqs):
    lengths = [len(seq) for seq in seqs]
    x = np.zeros((len(seqs), np.max(lengths))).astype('int32')
    for i, seq in enumerate(seqs):
        x[i, :lengths[i]] = seq
    return x


def get_mb_idxs(n, mb_size):
    mbs = []
    for i in np.arange(0, n, mb_size):
        mbs.append(np.arange(i, min(i+mb_size, n)))
    return mbs


def generate_batches(x, y, batch_size):
    """
        Divy examples into batches of given size
    """
    mbs = get_mb_idxs(len(x), batch_size)
    batches = []
    for mb in mbs:
        mb_x = [x[i] for i in mb]
        mb_y = [y[i] for i in mb]
        mb_x = pad_data(mb_x)
        batches.append((mb_x, mb_y))
    return batches
