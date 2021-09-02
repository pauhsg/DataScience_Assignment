import matplotlib.pyplot as plt
import nltk
import numpy as np
import re

from nltk.corpus import stopwords, wordnet
from nltk import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# dictionnary of contractions like "don't" that will be replaced by the the entire expression -> "do not"
'''
List adapted from:
https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
'''
CONTRACTIONS = {
    "ain't": "am not",
      "aren't": "are not",
      "can't": "can not",
      "can't've": "can not have",
      "'cause": "because",
      "could've": "could have",
      "couldn't": "could not",
      "couldn't've": "could not have",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "hadn't": "had not",
      "hadn't've": "had not have",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he would",
      "he'd've": "he would have",
      "he'll": "he will",
      "he'll've": "he will have",
      "he's": "he is",
      "how'd": "how did",
      "how'd'y": "how do you",
      "how'll": "how will",
      "how's": "how is",
      "i'd": "I would",
      "i'd've": "I would have",
      "i'll": "I will",
      "i'll've": "I will have",
      "i'm": "I am",
      "i've": "I have",
      "isn't": "is not",
      "it'd": "it had",
      "it'd've": "it would have",
      "it'll": "it will",
      "it'll've": "it will have",
      "it's": "it is",
      "let's": "let us",
      "ma'am": "madam",
      "mayn't": "may not",
      "might've": "might have",
      "mightn't": "might not",
      "mightn't've": "might not have",
      "must've": "must have",
      "mustn't": "must not",
      "mustn't've": "must not have",
      "needn't": "need not",
      "needn't've": "need not have",
      "o'clock": "of the clock",
      "oughtn't": "ought not",
      "oughtn't've": "ought not have",
      "shan't": "shall not",
      "sha'n't": "shall not",
      "shan't've": "shall not have",
      "she'd": "she would",
      "she'd've": "she would have",
      "she'll": "she will",
      "she'll've": "she will have",
      "she's": "she is",
      "should've": "should have",
      "shouldn't": "should not",
      "shouldn't've": "should not have",
      "so've": "so have",
      "so's": "so is",
      "that'd": "that would",
      "that'd've": "that would have",
      "that's": "that is",
      "there'd": "there had",
      "there'd've": "there would have",
      "there's": "there is",
      "they'd": "they would",
      "they'd've": "they would have",
      "they'll": "they will",
      "they'll've": "they will have",
      "they're": "they are",
      "they've": "they have",
      "to've": "to have",
      "wasn't": "was not",
      "we'd": "we had",
      "we'd've": "we would have",
      "we'll": "we will",
      "we'll've": "we will have",
      "we're": "we are",
      "we've": "we have",
      "weren't": "were not",
      "what'll": "what will",
      "what'll've": "what will have",
      "what're": "what are",
      "what's": "what is",
      "what've": "what have",
      "when's": "when is",
      "when've": "when have",
      "where'd": "where did",
      "where's": "where is",
      "where've": "where have",
      "who'll": "who will",
      "who'll've": "who will have",
      "who's": "who is",
      "who've": "who have",
      "why's": "why is",
      "why've": "why have",
      "will've": "will have",
      "won't": "will not",
      "won't've": "will not have",
      "would've": "would have",
      "wouldn't": "would not",
      "wouldn't've": "would not have",
      "y'all": "you all",
      "y'alls": "you alls",
      "y'all'd": "you all would",
      "y'all'd've": "you all would have",
      "y'all're": "you all are",
      "y'all've": "you all have",
      "you'd": "you had",
      "you'd've": "you would have",
      "you'll": "you you will",
      "you'll've": "you you will have",
      "you're": "you are",
      "you've": "you have"
}    
LINE_BREAKS = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NON_ALPHA = re.compile('[^A-Za-zÀ-ÿ]')
STOP_WORDS = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*') 


def expandContractions(text):
    # expands contractions like don't -> do not
    c_re = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))
    
    def replace(match):
        return CONTRACTIONS[match.group(0)]
    
    return c_re.sub(replace, text)


def clean_reviews(reviews):
    clean_reviews = list()
    for review in reviews:
        # convert into lower cases
        review = review.lower()
        
        # expand contractions like don't -> do not    
        review = expandContractions(review) 

        # remove line breaks
        review = LINE_BREAKS.sub(' ', review) 

        # remove urls
        review = re.sub(r"http\S+", "", review) 
        
        # remove all non-alphabetical characters 
        review = NON_ALPHA.sub(' ', review) 
        
        # remove empty reviews
        if not (review.isspace() or review == np.nan or not review):
            review = review 
        
        # substitute multiple whitespace with single whitespace 
        review = ' '.join(review.split())
        
        # remove stopwords
        review = STOP_WORDS.sub('', review)
        
        # remove words that are < 2 letters
        review = [token for token in review.split() if len(token) > 2] 
        review = ' '.join(review)

        # add to clean reviews list
        clean_reviews.append(review)
    
    return clean_reviews


def tokenize_and_lemmatize(reviews):
    lemmatizer = WordNetLemmatizer()
    tok_and_lem_reviews = list()
    
    for review in reviews:
        # tokenize each review
        tokens = word_tokenize(review)

        # lemmatize the obtained tokens
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # add to lemmatized reviews list
        tok_lem_review = ' '.join(lemmatized_tokens)
        tok_and_lem_reviews.append(tok_lem_review)

    return tok_and_lem_reviews


def get_reviews_length_histogram(data):
    pos_reviews_length = list()
    neg_reviews_length = list()

    half_idx = int(len(data)/2 - 1)

    for review in data[0:half_idx]:
        pos_reviews_length.append(len(review))

    for review in data[half_idx+1:]:
        neg_reviews_length.append(len(review))
    
    
    fig, ax = plt.subplots()
    ax.hist(pos_reviews_length, color='green', label='Positive reviews')
    ax.hist(neg_reviews_length, color='red', label='Negative reviews')
    ax.set(xlabel='Reviews length', ylabel='Frequency', title='Review length distribution')
    ax.set_xlim([0, 8000])
    ax.legend()
    plt.show()