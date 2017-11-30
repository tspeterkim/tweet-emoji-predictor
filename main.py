import numpy as np

import utils

def main():
    tweets, emojis = utils.load_data(max_example=100)
    word_dict = utils.build_dict(tweets)
    # embeddings = utils.generate_embeddings(word_dict, dim=50, pretrained_path='data/glove.twitter.27B.50d.txt')
    embeddings = utils.generate_embeddings(word_dict, dim=50, pretrained_path=None)

    x, y = utils.vectorize(tweets, emojis, word_dict)

    all_train = utils.generate_batches(x,y,batch_size=32)

    # Psuedo tra
    for epoch in range(10):
        np.random.shuffle(all_train)
        for idx, (mb_x, mb_y) in enumerate(all_train):
            print('#Examples = %d, max_seq_len = %d' % (len(mb_x), mb_x.shape[1]))

        print("----")



if __name__ == '__main__':
    main()
