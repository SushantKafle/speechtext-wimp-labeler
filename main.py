from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, AnnotationDataset
from config import Config
from model import WImpModel

def main(config):
    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_chars = load_vocab(config.chars_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars,
        lowercase=True, chars=True)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # create dataset
    dev   = AnnotationDataset(config.dev_filename, processing_word)
    test  = AnnotationDataset(config.test_filename, processing_word)
    train = AnnotationDataset(config.train_filename, processing_word)

    print ("Num. train: %d" % len(train))
    print ("Num. test: %d" % len(test))
    print ("Num. dev: %d" % len(dev))

    model = WImpModel(config, embeddings, ntags=config.nclass,
        nchars=len(vocab_chars))
    
    # build WImpModel
    model.build_graph()

    # train, evaluate and interact
    model.train(train, dev)
    model.evaluate(test)

if __name__ == "__main__":
    # create instance of config
    config = Config()
    main(config)
        