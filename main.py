import numpy as np
import utils

from gru_nn import GRU_Classifier
from lstm import LSTM

import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable

run_LSTM = False
run_BD_LSTM = False
run_GRU = True

global_epoch_num = 500
global_learning_rate = 1e-3

def main():
    tweets, emojis = utils.load_data(path='data/us_train', max_example=100)
    dev_tweets, dev_emojis = utils.load_data(max_example=100)

    word_dict = utils.build_dict(tweets)
    # embeddings = utils.generate_embeddings(word_dict, dim=50, pretrained_path='data/glove.twitter.27B.50d.txt')
    embeddings = utils.generate_embeddings(word_dict, dim=50, pretrained_path=None)

    x, y = utils.vectorize(tweets, emojis, word_dict)

    all_train = utils.generate_batches(x,y,batch_size=32)

    # set the parameters
    batch_size, input_size, hidden_size, output_size = 32, 50, 200, 20
    vocabulary_size = len(embeddings)

    dev_x, dev_y = utils.vectorize(dev_tweets, dev_emojis, word_dict)
    all_dev = utils.generate_batches(dev_x, dev_y, batch_size=32)

    if run_GRU:

        # initialize the model
        model = GRU_Classifier(vocabulary_size, input_size, hidden_size, output_size, batch_size)
        # model.word_embeddings.weight.data = torch.from_numpy(embeddings)
        model.word_embeddings.weight.data = torch.FloatTensor(embeddings.tolist())
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        epoch_num = 500
        it = 0
        best_dev_acc = 0

        

        # model training
        for epoch in range(epoch_num):
            np.random.shuffle(all_train)
            for idx, (mb_x, mb_y) in enumerate(all_train):
                print('#Examples = %d, max_seq_len = %d' % (len(mb_x), mb_x.shape[1]))
                mb_x = Variable(torch.from_numpy(np.array(mb_x, dtype=np.int64)), requires_grad=False)
                y_pred, _ = model(mb_x, len(mb_x))
                mb_y = Variable(torch.from_numpy(np.array(mb_y, dtype=np.int64)), requires_grad=False)
                loss = loss_function(y_pred, mb_y)
                print('epoch ', epoch, 'batch ', idx, 'loss ', loss.data[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                it += 1

                if it % 100 == 0: # every 100 updates, check dev accuracy
                    correct = 0
                    n_examples = 0
                    for idx, (d_x, d_y) in enumerate(all_dev):
                        n_examples += len(d_x)

                        d_x = Variable(torch.from_numpy(np.array(d_x, dtype=np.int64)), requires_grad=False)
                        _, y_pred = model(d_x, len(d_x))
                        y_pred = y_pred.data.numpy()
                        emoji_pred = np.argmax(y_pred, axis=1)

                        correct += np.sum((emoji_pred == d_y).astype(int))

                    dev_acc = correct / n_examples
                    print("Dev Accuracy: %f" % dev_acc)
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        print("Best Dev Accuracy: %f" % best_dev_acc)


    if run_LSTM:
        print("Running LSTM...")
        bd_lstm = BD_LSTM(input_size, hidden_size, num_layers, num_classes)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(bd_lstm.parameters(), lr=global_learning_rate)
        
        #train BD LSTM
        for epoch in range(num_epochs):
            np.random.shuffle(all_train)

            for idx, (mb_x, mb_y) in enumerate(all_train):
                print('#Examples = %d, max_seq_len = %d' % (len(mb_x), mb_x.shape[1]))
                mb_x = Variable(torch.from_numpy(np.array(mb_x, dtype=np.int64)), requires_grad=False)
                y_pred, _ = bd_lstm(mb_x, len(mb_x))
                mb_y = Variable(torch.from_numpy(np.array(mb_y, dtype=np.int64)), requires_grad=False)
                loss = loss_function(y_pred, mb_y)
                print('epoch ', epoch, 'batch ', idx, 'loss ', loss.data[0])


                optimizer = zero_grad()
                loss.backward()
                optimizer.step()


    if run_BD_LSTM:
        print("Running BD_LSTM")


if __name__ == '__main__':
    main()
