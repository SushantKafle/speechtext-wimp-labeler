import os, math
from general_utils import get_logger

class Config():
    def __init__(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.logger = get_logger(self.log_path)

    # location of the Word Improtance Corpus
    wimp_corpus = "--UPDATE--"

    # location of the Switchboard transcripts
    swd_transcripts = "--UPDATE--"

    # type of model
    model = "lstm_crf"
    opt_metric = "f-score"
    nclass = 6

    random_seed = 1000

    # general config
    output_path = "results/exp-1/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    confusion_mat = output_path + "confusion-mat.png"
    compare_predictions = output_path + "compare-predictions.csv"

    # embeddings
    dim = 300
    dim_char = 100
    glove_filename = "data/glove.6B/glove.6B.300d.txt"
    trimmed_filename = "data/glove.6B.300d.trimmed.npz"

    # dataset
    dev_filename = "data/testa.txt"
    test_filename =  "data/testb.txt"
    train_filename = "data/train.txt"
    
    # vocab
    words_filename = "data/words.txt"
    chars_filename = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs = 20
    dropout = 0.5
    batch_size = 20
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    nepoch_no_imprv = 7
    reload = False
    
    # model hyperparameters
    word_rnn_size = 300
    char_rnn_size = 100


    # some utility functions
    def ann2class(self, tag):
        tag = float(tag)
        if self.nclass == 6:
            if tag < 0.1:
                return 0
            return int(math.ceil(tag/0.2))
        elif self.nclass == 3:
            if tag < 0.3:
                return 0
            elif tag < 0.6:
                return 1
            return 2
        elif self.nclass == 2:
            if tag < 0.5:
                return 0
            return 1

    def class2ann(self, tag):
        tag = float(tag)
        if self.nclass == 6:
            return tag/5.
        elif self.nclass == 3:
            return ((tag + 1) * 0.3 - 0.1)
        elif self.nclass == 2:
            return 0.25 if tag == 0 else 0.75

    def digitize_labels(self, tags):
        return list(map(self.ann2class, tags))


