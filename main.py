import numpy as np
import utils
from gru_nn import GRU_Classifier
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable

def main():
    tweets, emojis = utils.load_data(max_example=100)
    word_dict = utils.build_dict(tweets)
    # embeddings = utils.generate_embeddings(word_dict, dim=50, pretrained_path='data/glove.twitter.27B.50d.txt')
    embeddings = utils.generate_embeddings(word_dict, dim=50, pretrained_path=None)

    x, y = utils.vectorize(tweets, emojis, word_dict)

    all_train = utils.generate_batches(x,y,batch_size=32)

    # set the parameters
    batch_size, embedding_dim, hidden_size, output_size = 32, 50, 200, 20
    vocabulary_size = len(embeddings)

    # initialize the model
    model = GRU_Classifier(vocabulary_size, embedding_dim, hidden_size, output_size)
    model.word_embeddings.weight.data = torch.from_numpy(embeddings)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    epoch_num = 500

    # model training
    for epoch in range(epoch_num):
        np.random.shuffle(all_train)
        for idx, (mb_x, mb_y) in enumerate(all_train):
            print('#Examples = %d, max_seq_len = %d' % (len(mb_x), mb_x.shape[1]))
            current_bs = len(mb_x)

            mb_x = Variable(torch.from_numpy(np.array(mb_x, dtype=np.int64)))
            y_pred = model(mb_x, current_bs)
            mb_y = Variable(torch.from_numpy(np.array(mb_y, dtype=np.int64)))
            loss = loss_function(y_pred, torch.from_numpy(mb_y))
            print('epoch ', epoch, 'batch ', idx, loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("----")


if __name__ == '__main__':
    main()
