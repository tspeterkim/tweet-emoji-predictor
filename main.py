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
    batch_size, input_size, hidden_size, output_size = 32, 50, 200, 20
    vocabulary_size = len(embeddings)

    # initialize the model
    model = GRU_Classifier(vocabulary_size, input_size, hidden_size, output_size, batch_size)
    # model.word_embeddings.weight.data = torch.from_numpy(embeddings)
    model.word_embeddings.weight.data = torch.FloatTensor(embeddings.tolist())
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    epoch_num = 500

    # model training
    for epoch in range(epoch_num):
        np.random.shuffle(all_train)
        for idx, (mb_x, mb_y) in enumerate(all_train):
            print('#Examples = %d, max_seq_len = %d' % (len(mb_x), mb_x.shape[1]))

            # mb_x = Variable(torch.from_numpy(np.array(mb_x, dtype=np.int64)))
            print("mb_x: ", np.shape(mb_x))
            mb_x = Variable(torch.LongTensor(np.array(mb_x, dtype=np.int64).tolist()))
            y_pred = model(mb_x, len(mb_x))
            # mb_y = Variable(torch.from_numpy(np.array(mb_y, dtype=np.int64)))
            mb_y = Variable(torch.LongTensor(mb_y))
            # batch_size * class_count
            loss = loss_function(y_pred, mb_y)
            print('epoch ', epoch, 'batch ', idx, loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("----")


if __name__ == '__main__':
    main()
