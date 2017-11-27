import utils

def main():
    tweets, emojis = utils.load_data(max_example=100)
    word_dict = utils.build_dict(tweets)
    embeddings = utils.generate_embeddings(word_dict, dim=100, pretrained_path='data/glove.6B.100d.txt')


if __name__ == '__main__':
    main()
