import utils

def main():
    tweets, emojis = utils.load_data(max_example=100)
    word_dict = utils.build_dict(tweets)
    # embeddings = utils.generate_embeddings(word_dict, dim=50, pretrained_path='data/glove.twitter.27B.50d.txt')
    embeddings = utils.generate_embeddings(word_dict, dim=50, pretrained_path=None)


if __name__ == '__main__':
    main()
