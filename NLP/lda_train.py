import numba
# to compile the code
from collections import Counter
import numpy as np
import tqdm
import pandas as pd
from typing import*
# to make more clear
from time import time
import random

class TopicModeling:

    def __init__(self, file=None, alpha:float=.1, beta:float=.1, k:int=10)->None:

            'alpha and beta the parameters that controls the dirichlet distributions'

            self.file=file
            assert isinstance(alpha, float)
            self._alpha=alpha
            assert isinstance(beta, float)
            self._beta=beta
            assert isinstance(k, int)
            self._k=k

    # getting the parameters and asserting their propre types
    @property
    def alpha(self)->float:
            assert isinstance(self._alpha, float)
            return self._alpha

    @property
    def beta(self)->float:
            assert isinstance(self._beta, float)
            return self._beta

    @property
    def k(self)->int:
            assert isinstance(self._k, int)
            return self._k

    # setting new paarameters
    @alpha.setter
    def alpha(self, new_val)->float:
            assert isinstance(new_val, float)
            self._alpha=new_val
            return self._alpha

    @beta.setter
    def alpha(self, new_val)->float:
            assert isinstance(new_val, float)
            self._beta = new_val
            return self._beta

    @k.setter
    def alpha(self, new_val)->int:
            assert isinstance(new_val, int)
            self._k = new_val
            return self._k

    @numba.jit
    @staticmethod
    def preprocessing(self):
        file=self.file
        data=pd.read_csv(self.file)

    @staticmethod
    def Gibbs_sampling():
        pass

    @staticmethod
    def choose_element(List:Optional[list, np.array()], weights:np.array())->int:
        """
        Chooses an element from a list based on the weights.

        Args:
          List: A list or numpy.array() of elements.
          weights: A list of weights for each element in list(numpy.array()).

        Returns:
          An element from List, chosen based on the weights.
        """

        # Normalize the weights.
        weights = weights / np.sum(weights)

        # Choose an element from List based on the weights.
        index = np.random.choice(len(List), p=weights)

        return List[index]

    @staticmethod
    def n_LDA(self)->List[np.array(), np.array(), np.array(), np.array()]:

        ''' to make code more efficient, we put the loops inside of np.array() or a list
        example:
        [ i for i in range(10)] more efficient than
        List=[]
        for i in range(10):
            List.append(i)
        '''

        # getting corpus from the data
        corpus=self.preprocessing()
        corpus_size=len(corpus)
        # Initialinzing randomly Z
        Z=[[random.randrange(self._k)] for d in corpus]
        ndk = np.zeros((corpus_size, self._k))
        # Incrementing the counts
        number_documents_topic= [[np.sum(Z[d] == k)for k in np.arange(self._k)] for d in corpus]
        number_topics_word = np.zeros((self._k, corpus_size))
        for d in range(corpus_size):
            for word , topic in zip(corpus[d], number_topics_word[d]):
                number_topics_word[d, topic]+=1
        # we sum with respect the words
        number_topics=np.sum(number_topics_word, axis=1)
        # the topics are latent (hidden) variables so we encoded them by [1, 2, ...]
        topics=np.arange(1, self._k+1)

        # after initializing all variables and parameters and increments the counts
        # it's time to apply lda
        for _ in tqdm.trange(self.num_iterations):
            # tqdm.trange for progress
            for d in range(corpus_size):
                for d, doc in  enumerate(zip(corpus[d], number_documents_topic[d])):
                    # remove the topic that conditioneed by this word
                    number_documents_topic[d, topic]-=1
                    number_topics_word[topic, word]-=1
                    number_topics[topic]-=1
                    p_z = (number_documents_topic[d, :] + self._alpha) * (number_topics_word[:, word] + self._beta) / (number_topics[:] + self._beta* corpus_size)
                    # sampling new topic by considering the weights
                    topic=self.choose_element(topics, weights=p_z)
                    Z[d, i]=topic
                    number_documents_topic[d, topic]+=1
                    number_topics_word[topic, word] += 1
                    number_topics[topic] += 1

        return Z, number_documents_topic, number_topics_word, number_topics

    @staticmethod
    def most_common_word(num_word):
        for k in range(self._k):
            most_common_word=np.sort()
