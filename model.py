import numpy as np
import os, csv, math
import tensorflow as tf
from data_utils import minibatches, pad_sequences
from general_utils import Progbar, plot_confusion_matrix
from model_utils import create_feedforward, get_rnn_cell
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix


class WImpModel(object):

    def __init__(self, config, embeddings, ntags, nchars = None):
        self.config     = config
        self.embeddings = embeddings
        self.nchars     = nchars
        self.ntags      = ntags
        self.logger     = config.logger
        self.rng        = np.random.RandomState(self.config.random_seed)

    
    def _init_graph_(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
            name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
            name="sequence_lengths")

        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
            name="char_ids")
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
            name="word_lengths")

        if self.config.model == "lstm_crf": 
            self.imp_labels = tf.placeholder(tf.int32, shape=[None, None], 
                name="imp_labels")
        else:
            self.imp_labels = tf.placeholder(tf.float32, shape=[None, None], 
                name="imp_labels")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
            name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], 
            name="lr")


    def create_initializer(self, size):
        return tf.constant(np.asarray(self.rng.normal(loc = 0.0, scale = 0.1, size = size), dtype = np.float32))


    def get_feed_dict(self, words, imp_labels = None, lr = None, dropout = None):
        char_ids, word_ids = zip(*words)
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        feed[self.char_ids] = char_ids
        feed[self.word_lengths] = word_lengths

        if imp_labels is not None:
            imp_labels, _ = pad_sequences(imp_labels, 0)
            feed[self.imp_labels] = imp_labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def _define_embedings_(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name = "_word_embeddings", dtype = tf.float32, 
                trainable = self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, 
                name = "word_embeddings")

        with tf.variable_scope("chars"):
            _char_embeddings = tf.get_variable(name = "_char_embeddings", dtype = tf.float32, 
                shape = [self.nchars, self.config.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, 
                name = "char_embeddings")
            
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings, shape = [-1, s[-2], self.config.dim_char])
            word_lengths = tf.reshape(self.word_lengths, shape = [-1])
            
            cell_fw = get_rnn_cell(self.config.char_rnn_size, "LSTM", state_is_tuple = True)
            cell_bw = get_rnn_cell(self.config.char_rnn_size, "LSTM", state_is_tuple = True)

            _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                cell_bw, char_embeddings, sequence_length = word_lengths, 
                dtype = tf.float32)

            output = tf.concat([output_fw, output_bw], axis = -1)
            output = tf.reshape(output, shape= [-1, s[1], 2 * self.config.char_rnn_size])

            word_embeddings = tf.concat([word_embeddings, output], axis=-1)
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def _define_logits_(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = get_rnn_cell(self.config.word_rnn_size, "LSTM")
            cell_bw = get_rnn_cell(self.config.word_rnn_size, "LSTM")
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                cell_bw, self.word_embeddings, sequence_length=self.sequence_lengths, 
                dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)

        ntime_steps = tf.shape(output)[1]
        output = tf.reshape(output, [-1, 2 * self.config.word_rnn_size])
        pred_dim = 1 if self.config.model == "lstm_sig" else self.ntags
        pred = create_feedforward(output, 2 * self.config.word_rnn_size, pred_dim, self.create_initializer,
        "linear", "projection")

        if self.config.model == "lstm_sig":
            pred = tf.sigmoid(pred)
            self.logits = tf.reshape(pred, [-1, ntime_steps])
        else:
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

    def _define_predictions_(self):
        self.imp_pred = self.logits


    def _define_loss_(self):
        if self.config.model == "lstm_sig":
            Y = tf.reshape(self.imp_labels, [-1, 1])
            pred_Y = tf.reshape(self.logits, [-1, 1])
            self.loss = tf.sqrt(tf.reduce_sum(tf.pow(pred_Y - Y, 2)))
        else:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.imp_labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)

    def _setup_optimizer_(self):
        with tf.variable_scope("optimizer_setup"):
            if self.config.lr_method == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.lr_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.config.lr_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                optimizer = tf.train.RMSPropOptimizer(self.lr)

            self.optimize_ = optimizer.minimize(self.loss)
        
    
    def build_graph(self):
        self._init_graph_()
        self._define_embedings_()
        self._define_logits_()
        self._define_predictions_()
        self._define_loss_()
        self._setup_optimizer_()
        
        self.init = tf.global_variables_initializer()


    def predict_batch(self, sess, words):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        if self.config.model == "lstm_sig":
            imp_pred = sess.run(self.imp_pred, feed_dict=fd)
        else:
            imp_pred = []
            logits, transition_params = sess.run([self.logits, self.transition_params], 
                    feed_dict=fd)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                logit, transition_params)
                imp_pred += [viterbi_sequence]
        return imp_pred, sequence_lengths


    def run_epoch(self, sess, train, dev, epoch):
        nbatches = (len(train) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (words, imp_labels) in enumerate(minibatches(train, self.config.batch_size)):

            if self.config.model == "lstm_crf":
                imp_labels = list(map(self.config.digitize_labels, imp_labels))

            fd, _ = self.get_feed_dict(words, imp_labels, self.config.lr, self.config.dropout)
            _, train_loss = sess.run([self.optimize_, self.loss], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])

        result = self.run_evaluate(sess, dev)
        self.logger.info("- dev acc {:04.4f} - f {:04.4f} - rms {:04.4f}".format(100*result['accuracy'], 
            100 * result['f-score'], -1 * result['rms']))
        return result

    def run_evaluate(self, sess, test, save=False):
        accs, rms = [], []
        labs, labs_ = [], []
        for words, imp_labels in minibatches(test, self.config.batch_size):
            imp_labels_, sequence_lengths = self.predict_batch(sess, words)
            for lab, lab_, length in zip(imp_labels, imp_labels_, sequence_lengths):
                lab = lab[:length]
                lab_ = lab_[:length]

                if self.config.model == "lstm_sig":
                    d_lab = map(self.config.ann2class, lab)
                    d_lab_ = map(self.config.ann2class, lab_)
                else:
                    d_lab = list(map(self.config.ann2class, lab))
                    d_lab_ = lab_[:]
                    lab_ = list(map(self.config.class2ann, d_lab_))

                rms += [pow((float(a)-float(b)), 2) for (a,b) in zip(lab, lab_)]
                accs += [a==b for (a, b) in zip(d_lab, d_lab_)]

                labs.extend(d_lab)
                labs_.extend(d_lab_)

        if save:
            with open(self.config.compare_predictions, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['truth', 'predictions'])
                for y, pred_y in zip(labs, labs_):
                    csv_writer.writerow([y, pred_y])
            print ("'compare.csv' file saved!")

        p, r, f, s = score(labs, labs_, average="macro")
        cnf_mat = confusion_matrix(labs, labs_)
        acc = np.mean(accs)
        rms_ = np.sqrt(np.mean(rms))
        return {'accuracy': acc, 'precision': p, 'recall': r, 'f-score': f, 'cnf': cnf_mat, 'rms': -1 * rms_}


    def train(self, train, dev):
        best_score = -100
        saver = tf.train.Saver()
        
        nepoch_no_imprv = 0
        with tf.Session() as sess:
            sess.run(self.init)

            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
                result = self.run_epoch(sess, train, dev, epoch)

                self.config.lr *= self.config.lr_decay

                if result[self.config.opt_metric] >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = result[self.config.opt_metric]
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                                        nepoch_no_imprv))
                        break


    def evaluate(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.config.model_output)
            result = self.run_evaluate(sess, test, save=True)

            #plot the confustion matrix
            plot_confusion_matrix(self.config, result['cnf'], classes=[str(i) for i in range(0, 6)], normalize=True,
                      title='Normalized confusion matrix')
            self.logger.info("- test acc {:04.4f} - f {:04.4f} - rms {:04.4f}".format(100 * result['accuracy'], 
                100 * result['f-score'], -1 * result['rms']))


    def annotate_files(self, file_path, out_path, processing_word):
        output_file = open(out_path, 'w')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.config.model_output)
            with open(file_path) as f:
                for line in f:
                    sentence = line.strip()

                    if sentence == "":
                        output_file.write("<empty> 0\n")
                        output_file.write("\n")
                    else:
                        words_raw = sentence.strip().split(" ")

                        words = [processing_word(w) for w in words_raw]
                        if type(words[0]) == tuple:
                            words = zip(*words)
                        preds, _ = self.predict_batch(sess, [words])
                        preds = preds[0]

                        for w, pred in zip(words_raw, preds):
                            output_file.write(w + " " + str(pred) + "\n")
                        output_file.write("\n")
        output_file.close()


    def reset(self):
        tf.reset_default_graph()
