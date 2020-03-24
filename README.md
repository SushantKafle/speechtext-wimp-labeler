Generic Bi-LSTM Model for Word Importance Prediction in Spoken Dialogues:
==============================================================

This project demonstrates the use of generic bi-directional LSTM models for predicting importance of words in a spoken dialgoue for understanding its meaning. The model operates on human-annotated corpus of word importance for its training and evaluation. The corpus can be downloaded from: http://latlab.ist.rit.edu/lrec2018

![Word Importance Visualization in a Dialgoue](https://github.com/SushantKafle/speechtext-wimp-labeler/blob/master/images/word-importance.png "Word Importance Visualization in a Dialgoue")

Performance Summary (will-be-updated-soon):
-------------------------------------------
<br/>

| Model     | Classes   | f-score   | rms   |
|:------------------------------:   |:-------:  |:-------:  |------ |
| bi-LSTM (char-embedding + CRF)    | 3     | 0.73    | 0.598  |
| bi-LSTM (char-embedding + CRF)    | 6     | 0.60    | 0.154  |

<br/>

You can cite this work and/or the corpus using:

> Sushant Kafle and Matt Huenerfauth. 2018.  A Corpus for  Modeling  Word  Importance  in  Spoken  Dialogue Transcripts.   In Proceedings  of  the  11th  edition  of the Language Resources and Evaluation Conference (LREC). ACM.


I. Data Preparation
====================

1. Download and locate the text transcripts of the Switchboard Corpus and the corresponding word importance corpus:
	*	The word importance corpus can be downloaded from this website: http://latlab.ist.rit.edu/lrec2018
	*	The text transcripts form the Switchboard corpus can be downloaded via this link: https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz

	*	In the “config.py” file, update the varibles shown below:

	    	# location of the Word Importance Corpus "annotations folder"
	    	wimp_corpus = --HERE-- 

	    	# location of the Switchboard transcripts
	    	swd_transcripts = --AND HERE--

2. Download glove vectors `glove.6B.300d.txt` from http://nlp.stanford.edu/data/glove.6B.zip and update `glove_filename` in `config.py`

3. Run the ‘build_data.py’ to prepare data for training, development and testing as:

	```python build_data.py```

This will create all the necessary files (such as the word vocabulary, character vocabulary and the training, development and test files) in the “$PROJECT_HOME/data/“ directory.

II. Install Python Dependencies
======================

`pip install -r requirements.txt`


III. Running the model
======================

1. Traverse inside the model you want to train and open the ‘config.py’ file and review the configurations:

	*	model : type of model to run (options: lstm_crf or lstm_sig)
	*	wimp_corpus : Path to the Word Importance Corpus
	*	swd_transcripts : Path to the Switchboard Transcripts
	*	output_path :	Path to the output directory
    *	model_output :	Path to save the best performing model
    *	log_path :	Path to store the log
    *	confusion_mat :	Path to save the image of the confusion matrix (part of analysis on the test data)
    *	compare_predictions : Path to save the predictions of the model (.csv file is produced)

    *   random_seed :  Random seed
    *   opt_metric : Metric to evaluate the progress at each epoch
    *   nclass : Num of classes for prediction

    *	dim : Size of the word embeddings used in the model
    *	dim_char : Size of the character embeddings
    *	glove_filename : Path to the glove-embeddings file
    *	trimmed_filename : Path to save the trimmed glove-embeddings

    *	dev_filename : Path to the development data, used to select the best epoch
    *	test_filename : Path to the test data, use for evaluating the model performance
    *	train_filename : Path to the train data

    *	words_filename : Path to the word vocabulary
    *	tags_filename : Path to the vocabulary of the tags
    *	chars_filename : Path to the vocabulary of the characters

    *	train_embeddings : If True, trains the word-level embeddings
    *	nepochs : Maximum number of epoches to run
    *	dropout : The probability of applying dropout during training
    *	batch_size : Number of examples in each batch
    *	lr_method : Optimization strategy (options: adam, adagrad, sgd, rmsprop)
    *	lr : Learning rate
    *	lr_decay : Rate of decay of the learning rate
    *	clip : Gradient clipping, if negative no clipping
    *	nepoch_no_imprv : Number of epoch without improvement for early termination
    *	reload : Reload the latest trained model

    *	word_rnn_size : Size of the word-level LSTM hidden layers
    *	char_rnn_size : Size of the char-level LSTM hidden layers

2. Run the model by:

	```python main.py```

Summary:
*	trained model saved at “model_output” (declared inside config.py).
*	log of the analysis at “log_path” (declared inside config.py) - contains train, dev and test performance.
*	confusion matrix at “confusion_mat” (declared inside config.py).
*	CSV file containing the actual scores and the predicted score at “compare_predictions” (declared inside config.py).


IV. Running the agreement analysis
====================================

1. Locate the csv file containing the actual scores annotated by the annotators and the predicted scores.

2. Open up “compare-annotation.R” file and update the “annotation_src” variable with this new location.
