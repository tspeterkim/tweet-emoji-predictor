import numpy as np
from sklearn.metrics import f1_score
import utils

from gru import GRU_Classifier
from lstm import LSTM_Classifier

import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable

import os.path
from timeit import default_timer as timer

run_LSTM = True
run_GRU = False
run_BD_GRU = False
run_BD_LSTM = False

global_epoch_num = 500
global_learning_rate = 1e-3
max_example = 50
max_dev_example = 50

def main():

    start = timer()

    if(os.path.isfile("data/tweets"+str(max_example)+".npy") and os.path.isfile("data/emojis"+str(max_example)+".npy")):
        tweets = np.load("data/tweets"+str(max_example)+".npy").tolist()
        emojis = np.load("data/emojis"+str(max_example)+".npy").tolist()
    else:
        tweets, emojis = utils.load_data(path='data/us_train', max_example=max_example)
        np.save("data/tweets"+str(max_example)+".npy", np.array(tweets))
        np.save("data/emojis"+str(max_example)+".npy", np.array(emojis))

    if(os.path.isfile("data/dev_tweets"+str(max_dev_example)+".npy") and os.path.isfile("data/dev_emojis"+str(max_dev_example)+".npy")):
        dev_tweets = np.load("data/dev_tweets"+str(max_dev_example)+".npy").tolist()
        dev_emojis = np.load("data/dev_emojis"+str(max_dev_example)+".npy").tolist()
    else:
        dev_tweets, dev_emojis = utils.load_data(max_example=max_dev_example)
        np.save("data/dev_tweets"+str(max_dev_example)+".npy", np.array(dev_tweets))
        np.save("data/dev_emojis"+str(max_dev_example)+".npy", np.array(dev_emojis))

    start1 = timer()
    print(start1-start)

    word_dict = utils.build_dict(tweets)
    # embeddings = utils.generate_embeddings(word_dict, dim=300, pretrained_path='data/glove.6B.300d.txt')
    embeddings = utils.generate_embeddings(word_dict, dim=300, pretrained_path=None)

    end0 = timer()
    print(end0-start1)

    x, y = utils.vectorize(tweets, emojis, word_dict)
    dev_x, dev_y = utils.vectorize(dev_tweets, dev_emojis, word_dict)

    end1 = timer()
    print(end1-end0)

    batch_size, input_size, hidden_size, output_size, layers = 32, 300, 200, 20, 1
    all_train = utils.generate_batches(x,y,batch_size=batch_size)
    all_dev = utils.generate_batches(dev_x, dev_y, batch_size=batch_size)

    end2 = timer()
    print(end2-end1)

    # set the parameters
    # batch_size, input_size, hidden_size, output_size, layers = 64, 50, 200, 20, 1
    vocabulary_size = len(embeddings)

    if run_GRU:
        print("running GRU...")
        # initialize the model
        model = GRU_Classifier(vocabulary_size, input_size, hidden_size, output_size, layers, run_BD_GRU)
        model.word_embeddings.weight.data = torch.FloatTensor(embeddings.tolist())
        if torch.cuda.is_available():
            model.cuda()
            (model.word_embeddings.weight.data).cuda()


        loss_function = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_function.cuda()

        optimizer = optim.Adam(model.parameters(), lr=global_learning_rate)
        epoch_num = 500
        it = 0
        best_dev_acc = 0
        best_f1 = 0


        # model training
        for epoch in range(epoch_num):
            np.random.shuffle(all_train)
            for idx, (mb_x, mb_y, mb_lengths) in enumerate(all_train):
                # sort the input in descending order according to sentence length
                # This is required by nn.utils.rnn.pack_padded_sequence
                sorted_index = len_value_argsort(mb_lengths)
                mb_x = [mb_x[i] for i in sorted_index]
                mb_y = [mb_y[i] for i in sorted_index]
                mb_lengths = [mb_lengths[i] for i in sorted_index]

                print('#Examples = %d, max_seq_len = %d' % (len(mb_x), len(mb_x[0])))
                mb_x = Variable(torch.from_numpy(np.array(mb_x, dtype=np.int64)), requires_grad=False)
                if torch.cuda.is_available():
                    mb_x = mb_x.cuda()

                y_pred = model(mb_x.t(), mb_lengths)
                mb_y = Variable(torch.from_numpy(np.array(mb_y, dtype=np.int64)), requires_grad=False)
                if torch.cuda.is_available():
                    mb_y = mb_y.cuda()
                loss = loss_function(y_pred, mb_y)
                # print('epoch ', epoch, 'batch ', idx, 'loss ', loss.data[0])

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                it += 1

                if it % 100 == 0: # every 100 updates, check dev accuracy
                    correct = 0
                    n_examples = 0
                    ground_truth = []
                    predicted = []
                    for idx, (d_x, d_y, d_lengths) in enumerate(all_dev):
                        ground_truth += d_y
                        n_examples += len(d_x)

                        sorted_index = len_value_argsort(d_lengths)
                        d_x = [d_x[i] for i in sorted_index]
                        d_y = [d_y[i] for i in sorted_index]
                        d_lengths = [d_lengths[i] for i in sorted_index]

                        d_x = Variable(torch.from_numpy(np.array(d_x, dtype=np.int64)), requires_grad=False)
                        if torch.cuda.is_available():
                            d_x = d_x.cuda()

                        # use pytorch way to calculate the correct count
                        d_y = Variable(torch.from_numpy(np.array(d_y, dtype=np.int64)), requires_grad=False)
                        if torch.cuda.is_available():
                            d_y = d_y.cuda()
                        y_pred = model(d_x.t(), d_lengths)
                        predicted += list(torch.max(y_pred, 1)[1].view(d_y.size()).data)
                        correct += (torch.max(y_pred, 1)[1].view(d_y.size()).data == d_y.data).sum()

                    dev_acc = correct / n_examples
                    f1 = f1_score(ground_truth, predicted, average='macro')
                    print("Dev Accuracy: %f, F1 Score: %f" % (dev_acc, f1))
                    if f1 > best_f1:
                        best_f1 = f1
                        print("Best F1 Score: %f" % best_f1)
                        gru_output = open('./out/gru_best', 'w')
                        gru_output.write(str(ground_truth) + '\n')
                        gru_output.write(str(predicted) + '\n')
                        gru_output.write(str(best_f1) + ' ' + str(dev_acc))
                        gru_output.close()

                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        print("Best Dev Accuracy: %f" % best_dev_acc)

    if run_LSTM:
        print("Running LSTM...")
        model = LSTM_Classifier(vocabulary_size, input_size, hidden_size, output_size, layers, run_BD_LSTM)
        model.word_embeddings.weight.data = torch.FloatTensor(embeddings.tolist())
        if torch.cuda.is_available():
            model.cuda()
            (model.word_embeddings.weight.data).cuda()

        loss_function = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_function.cuda()

        optimizer = optim.Adam(model.parameters(), lr=global_learning_rate)
        it = 0
        best_dev_acc = 0
        best_f1 = 0
        epoch_num = 500

        # train LSTM
        for epoch in range(epoch_num):
            np.random.shuffle(all_train)

            for idx, (mb_x, mb_y, mb_lengths) in enumerate(all_train):
                sorted_index = len_value_argsort(mb_lengths)
                mb_x = [mb_x[i] for i in sorted_index]
                mb_y = [mb_y[i] for i in sorted_index]
                mb_lengths = [mb_lengths[i] for i in sorted_index]
                print('#Examples = %d, max_seq_len = %d' % (len(mb_x), len(mb_x[0])))

                mb_x = Variable(torch.from_numpy(np.array(mb_x, dtype=np.int64)), requires_grad=False)
                if torch.cuda.is_available():
                    mb_x = mb_x.cuda()

                y_pred = model(mb_x.t(), mb_lengths)
                mb_y = Variable(torch.from_numpy(np.array(mb_y, dtype=np.int64)), requires_grad=False)
                if torch.cuda.is_available():
                    mb_y = mb_y.cuda()

                loss = loss_function(y_pred, mb_y)
                # print('epoch ', epoch, 'batch ', idx, 'loss ', loss.data[0])

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                it += 1


                if it % 100 == 0: # every 100 updates, check dev accuracy
                    correct = 0
                    n_examples = 0
                    ground_truth = []
                    predicted = []
                    for idx, (d_x, d_y, d_lengths) in enumerate(all_dev):
                        ground_truth += d_y
                        n_examples += len(d_x)

                        sorted_index = len_value_argsort(d_lengths)
                        d_x = [d_x[i] for i in sorted_index]
                        d_y = [d_y[i] for i in sorted_index]
                        d_lengths = [d_lengths[i] for i in sorted_index]

                        d_x = Variable(torch.from_numpy(np.array(d_x, dtype=np.int64)), requires_grad=False)
                        if torch.cuda.is_available():
                            d_x = d_x.cuda()

                        d_y = Variable(torch.from_numpy(np.array(d_y, dtype=np.int64)), requires_grad=False)
                        if torch.cuda.is_available():
                            d_y = d_y.cuda()
                        y_pred = model(d_x.t(), d_lengths)
                        predicted += list(torch.max(y_pred, 1)[1].view(d_y.size()).data)
                        correct += (torch.max(y_pred, 1)[1].view(d_y.size()).data == d_y.data).sum()

                    dev_acc = correct / n_examples
                    f1 = f1_score(ground_truth, predicted, average='macro')
                    print("Dev Accuracy: %f, F1 Score: %f" % (dev_acc, f1))
                    if f1 > best_f1:
                        best_f1 = f1
                        print("Best F1 Score: %f" % best_f1)
                        lstm_output = open('./out/lstm_best', 'w')
                        lstm_output.write(str(ground_truth) + '\n')
                        lstm_output.write(str(predicted) + '\n')
                        lstm_output.write(str(best_f1) + ' ' + str(dev_acc))
                        lstm_output.close()

                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        print("Best Dev Accuracy: %f" % best_dev_acc)


def len_value_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: seq[x], reverse=True)



if __name__ == '__main__':
    main()
