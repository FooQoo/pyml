from keras.datasets import imdb

def main():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         seed=113,
                                                         start_char=2,
                                                         oov_char=1,
                                                         index_from=0)

    i2w = {w_id: w for w, w_id in imdb.get_word_index().items()}

    with open('train.csv', mode='w', encoding='utf-8') as f:
        for i in range(x_train.shape[0]):
            line = ' '.join([i2w[w_id] for w_id in x_train[i]][1:])
            f.write('{0}, {1}\n'.format(line, y_train[i]))

    with open('test.csv', mode='w', encoding='utf-8') as f:
        for i in range(x_test.shape[0]):
            line = ' '.join([i2w[w_id] for w_id in x_test[i]][1:])
            f.write('{0}, {1}\n'.format(line, y_test[i]))
    

if __name__ == "__main__":
    main()