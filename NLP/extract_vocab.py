import numpy as np
import random
import spacy as sp
from collections import Counter

# reproducibility
def generate_frequencies(data, max_docs=10000):
    freqs=Counter()
    all_stopwords=sp.Defaults.stop_words
    nr_tokens=0
    for doc in data[:max_docs]:
        tokens=sp.tokenizer(doc)
        for token in tokens:
            token_text=token.lower()
            if token_text not in all_stopwords and  token.is_alpha:
                nr_tokens+=1
                freqs[token_text]+=1
    return freqs

def get_vocab(freqs, freq_threshold=3):
    vocab={}
    vocab_ixx_str={}
    vocab_idx={}
    for word in freqs:
        if freqs[word]>=freq_threshold:
            vocab[word]=vocab_idx
            vocab_ixx_str[vocab_idx]=word
            vocab_idx+=1
    return vocab, vocab_ixx_str
