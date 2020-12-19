from collections import defaultdict, Counter
from typing import List
import math
import numpy as np
from operator import itemgetter

class KneserNeyLM:

    def __init__(self, d=0.75):
        """
        KneserNey Language Modeling

        Args:
            d (int): the discounting parameter. We will fix it to 0.75 in this assignment
        """

        # discount parameter
        self.d = d

    def fit(self, sentences: List[str]) -> None:
        """
        Estimate the language model from training sentences

        Args:
            sentences (List[str]): a list of sentence. Each sentence is a str.
            <s> and </s> are not included in the sentences, make sure you add both <s> and </s> to each sentence

        Returns:
        """
        self.bigrams = {}
        self.unigrams = {}
        self.trigrams = {}
        self.num_wordi2_wordi1_star = {}
        self.num_star_wordi1_wordi = {}
        self.num_star_wordi1_star = {}
        self.num_wordi1_star = {}
        self.num_star_wordi = {}
        self.num_star_star_wordi = {}
        self.vocab = set()
        for line in sentences:
            words = line.split(' ')
            self.vocab.update(words)
        # Compute bigram and trigram frequencies
            for i, word in enumerate(words):
                self.unigrams[word] = self.unigrams.get(word, 0) + 1
                if i == 0:
                    self.bigrams['<S>', word] = self.bigrams.get(
                        ('<S>', word), 0) + 1
                    self.trigrams['<S>', '<S>', word] = self.trigrams.get(
                        ('<S>', '<S>', word), 0) + 1

                if i > 0:
                    self.bigrams[words[i-1],
                                 word] = self.bigrams.get((words[i-1], word), 0) + 1
                if i == 1:
                    self.trigrams['<S>', words[i-1],
                                  word] = self.trigrams.get(('<S>', words[i-1], word), 0) + 1
                if i > 1:
                    self.trigrams[words[i-2], words[i-1],
                                  word] = self.trigrams.get((words[i-2], words[i-1], word), 0) + 1
            self.bigrams[words[len(
                words) - 1], '</S>'] = self.bigrams.get((words[len(words) - 1], '</S>'), 0) + 1
            self.trigrams[words[len(words) - 2], words[len(words) - 1], '</S>'] = self.trigrams.get(
                (words[len(words) - 2], words[len(words) - 1], '</S>'), 0) + 1
            self.trigrams[words[len(words) - 1], '</S>', '</S>'] = self.trigrams.get(
                (words[len(words) - 1], '</S>', '</S>'), 0) + 1
            self.unigrams['<S>'] = self.unigrams.get('<S>', 0) + 1
            self.unigrams['</S>'] = self.unigrams.get('</S>',0) + 1

        for key in self.trigrams:
            self.num_wordi2_wordi1_star[key[0], key[1], '<star>'] = self.num_wordi2_wordi1_star.get(
                (key[0], key[1], '<star>'), 0) + 1
            self.num_star_wordi1_wordi['<star>', key[1], key[2]] = self.num_star_wordi1_wordi.get(
                ('<star>', key[1], key[2]), 0) + 1
            self.num_star_wordi1_star['<star>', key[1], '<star>'] = self.num_star_wordi1_star.get(
                ('<star>', key[1], '<star>'), 0) + 1
            self.num_star_star_wordi['<star>', '<star>', key[2]] = self.num_star_star_wordi.get(
                ('<star>', '<star>', key[2]), 0) + 1

        for key in self.bigrams:
            self.num_wordi1_star[key[0], '<star>'] = self.num_wordi1_star.get(
                (key[0], '<star>'), 0) + 1
            self.num_star_wordi['<star>', key[1]] = self.num_star_wordi.get(
                ('<star>', key[1]), 0) + 1


    def score_sent(self, sent):
        sent = (('<S>',) * (1) + sent + ('</S>',))
        sent_logprob = self.predict_proba_bigram(list(sent[0:2]))
        for i in range(0,len(sent) - 3 + 1):
            
            ngram = sent[i:i+3]
            sent_logprob *= self.logprob(ngram)
        return sent_logprob

    def logprob(self, ngram):
        return self.predict_proba_trigram(list(ngram))
    def replace_star(self, val):
        if val == '<s>':
            return '<S>'
        if val == '</s>':
            return '</S>'
        return val
    def predict_proba_unigram(self, unigram: List[str]) -> float:
        """
        Compute the simple count based probability P(w_1),
        return a small number if w_1 is not appearing in the dataset

        Args:
            unigram ([List[str])): one word

        Returns:
        """

        wordi = unigram[0]
        if wordi == '<s>':
            wordi = '<S>'
        if wordi == '</s>':
            wordi = '</S>'
        sum_vals = sum(v for v in self.unigrams.values())
        p = self.unigrams.get(wordi, 0) / sum_vals
        return p if p > 0 else 1/(len(self.vocab) + 2)


    def predict_proba_bigram(self, bigram: List[str]) -> float:
        """
        Compute the KneserNey bigram probability P(w_2|w_1)

        Args:
            bigram (List[str]): two words: [w_1, w_2]

        Returns: probability
        """
        wordi1 = self.replace_star(bigram[0])
        wordi = self.replace_star(bigram[1])

        if (self.num_wordi1_star.get((wordi1, '<star>'), 0)) == 0:
            p = 1
            return p

        denominator = self.num_star_wordi1_star.get(
            ('<star>', wordi1, '<star>'), 0)
        if denominator == 0:
            p = self.predict_proba_unigram(wordi)
            return p

        p_cont = self.num_star_wordi.get(('<star>', wordi), 0)  / len(self.bigrams.keys())

        p = (max(0, self.bigrams.get((wordi1, wordi), 0) - self.d)/self.unigrams.get(wordi1)) + (self.d * self.num_wordi1_star.get(
            (wordi1, '<star>')) / self.unigrams.get(wordi1) * (p_cont))
        return p


    def predict_proba_trigram(self, trigram: List[str]) -> float:
        """
        Compute the KneserNey trigram probability P(w_3|w_1, w_2)

        Args:
            bigram (List[str]): three words: [w_1, w_2, w_3]

        Returns: probability
        """
        wordi = self.replace_star(trigram[2])
        wordi1 = self.replace_star(trigram[1])
        wordi2 = self.replace_star(trigram[0])
        if (self.num_wordi2_wordi1_star.get((wordi2, wordi1, '<star>'), 0)) == 0:
            p = 1
            return p

        denominator = self.bigrams.get((wordi2, wordi1), 0)
        if denominator == 0:
            p = self.predict_proba_bigram([wordi1, wordi])
            return p

        # Kneser Ney trigram probability
        
        p_cont = self.num_star_wordi.get(('<star>', wordi), 0) / len(self.bigrams.keys())
        
        p = (max(0, self.trigrams.get((wordi2, wordi1, wordi), 0) - self.d)/self.bigrams.get((wordi2, wordi1), 0)) + (self.d * self.num_wordi2_wordi1_star.get((wordi2, wordi1, '<star>'), 0) /
                                                                                                                   self.bigrams.get((wordi2, wordi1), 0)) * (max(self.num_star_wordi1_wordi.get(('<star>', wordi1, wordi), 0)-self.d, 0) / self.num_star_wordi1_star.get(('<star>', wordi1, '<star>'), 0) + (self.d * self.num_wordi1_star.get((wordi1, '<star>'), 0)/ self.num_star_wordi1_star.get(('<star>', wordi1, '<star>'), 0) * p_cont))

        return p


    def perplexity(self, sentence: str) -> float:
        """
        compute perplexity for each sentence

        Args:
            sentence (str): a sentence. <s> and </s> are not included in the sentence

        Returns:
        """
        n = len(sentence.split(' '))
        log_prob = self.score_sent(tuple(sentence.split(" ")))
        print(f"Perplexity: {math.pow(1/log_prob, 1.0/n)}")
        return math.pow(1/log_prob, 1.0/n)



if __name__ == "__main__":
    obj = KneserNeyLM()
    sentences = []
    f = open('wsj.txt', 'r')
    for x in f:
        sentence = " ".join(x.strip().split(' ')[1:])
        sentences.append(sentence)
    #sentences = ['a a b c b a', 'b a c a b c c c b a', 'c b c a a a c a a b b a', 'b a c']
    #print(sentences[0])
    obj.fit(sentences)
    perplexities = []
    perplexity = 0
    #print(obj.perplexity("GENERAL"))
