__author__ = 'liam'
# -*- coding: utf-8 -*-

import pandas as pd
from kaggle import KaggleCompetition
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec
import logging


def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #
    # 5. Return a list of words
    return words


class Word2VecCompetition(KaggleCompetition):
    __full_name__ = 'word2vec-nlp-tutorial'
    __short_name__ = 'word2vec'
    __data_path__ = 'data'

if __name__ == '__main__':
    train_labeled = pd.read_table('data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3, index_col='id')
    train_unlabeled = pd.read_table('data/unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3, index_col='id')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []
    for i, review in enumerate(train_labeled.review):
        if i%100==0:
            print(i)
        sentences += review_to_sentences(review, tokenizer)
    for i, review in enumerate(train_unlabeled.review):
        if i%100==0:
            print(i)
        sentences += review_to_sentences(review, tokenizer)
    # reviews = labeled.review.apply(lambda text: review_to_wordlist(text, True))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                     sample=downsampling, seed=1)