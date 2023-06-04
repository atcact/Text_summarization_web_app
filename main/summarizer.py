from flask import Flask, request, jsonify
import math
import re
import sys
import numpy as np
import xml.etree.ElementTree as et
import networkx as nx

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm


class Summarizer:
    """
    The Summarizer class, which takes in a wiki and processes the documents in it
    into files that are used by the querier
    """

    def __init__(self, doc: str):

        # set of stop words
        self.STOP_WORDS = set(stopwords.words("english"))
        # porter stemmer
        self.nltk_ps = PorterStemmer()

        self.tokenization_regex = r"[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+"

        # sentence id to word to num appearances
        self.words_to_doc_frequency = {}
        # sentence id to number of words in sentence
        self.ids_to_max_counts = {}

        self.doc = doc
        self.shorten_num = 0
        self.sent_list = []
        self.run()


    def run(self):
        try:
            self.sent_list = self.sentence_tokenization(self.doc)
            self.shorten_num = (int)(len(self.sent_list) / 5)
            self.doc = self.process_document(self.doc)
            self.words_to_doc_relevance = self.compute_tf_idf_matrix()

        except FileNotFoundError:
            print("One (or more) of the files were not found")
        except IOError:
            print("Error: IO Exception")

    def stem_and_stop(self, word: str):
        """
        Checks if word is a stop word, converts it to lowercase, and stems it

        Parameters:
            word        the word to check
        Returns:
            "" if the word is a stop word, the converted word, otherwise
        """
        if word.lower() in self.STOP_WORDS:
            return ""

        return self.nltk_ps.stem(word.lower())

    def process_document(self, doc: str):
        token_list = []
        for word in re.findall(self.tokenization_regex, doc):
            stem_word = self.stem_and_stop(word)
            if not stem_word == "":
                token_list.append(stem_word)

        doc = ""
        for word in token_list:
            doc += word + " "

        return doc

    def sentence_tokenization(self, doc: str):
        list = re.findall(r"[^.!?]+", doc)
        sent_list = []
        for id, sent in enumerate(list):
            new_sent = ""
            sent = sent.split()
            for word in sent:
                stemmed = self.stem_and_stop(word)
                if not stemmed == "":
                    if id not in self.ids_to_max_counts:
                        self.ids_to_max_counts[id] = 1
                    else:
                        self.ids_to_max_counts[id] += 1
                    if stemmed not in self.words_to_doc_frequency:
                        self.words_to_doc_frequency[stemmed] = {id: 1}
                    elif stemmed in self.words_to_doc_frequency and id not in self.words_to_doc_frequency[stemmed]:
                        self.words_to_doc_frequency[stemmed][id] = 1
                    else: 
                        self.words_to_doc_frequency[stemmed][id] += 1

                    if new_sent == "":
                        new_sent = stemmed
                    else:
                        new_sent += " " + stemmed
            

            if not new_sent == "":
                sent_list.append(new_sent)
        return sent_list
                



    def compute_tf(self):
        """
        Computes tf metric based on words_to_doc frequency

        """
        
        tf_matrix = np.zeros(shape=(len(self.ids_to_max_counts),len(self.words_to_doc_frequency)))
    
        for i in range(len(self.ids_to_max_counts)):
            for j, word in enumerate(self.words_to_doc_frequency.keys()):
                if i in self.words_to_doc_frequency[word]:
                    tf_matrix[i][j] = self.words_to_doc_frequency[word][i] / self.ids_to_max_counts[i]
                else:
                    tf_matrix[i][j] = 0
        return tf_matrix

    def compute_idf(self):
        """
        Computes idf metric based on words_to_doc_frequency

        """

        idf_matrix = np.zeros(shape=(len(self.ids_to_max_counts),len(self.words_to_doc_frequency)))
        length = len(self.ids_to_max_counts)
        
        for j, word in enumerate(self.words_to_doc_frequency):
            sents_with_word_count = 0
            for i in self.ids_to_max_counts:
                if i in self.words_to_doc_frequency[word] and self.words_to_doc_frequency[word][i] > 0:
                    sents_with_word_count += 1
            for i in range(len(self.ids_to_max_counts)):
                idf_matrix[i][j] = math.log(length/sents_with_word_count)
        return idf_matrix


    def compute_tf_idf_matrix(self):
        tf_idf_matrix = np.zeros(shape=(len(self.ids_to_max_counts),len(self.words_to_doc_frequency)))
        tf_matrix = self.compute_tf()
        idf_matrix = self.compute_idf()
        for i in range(len(self.ids_to_max_counts)):
            for j in range(len(self.words_to_doc_frequency)):
                tf_idf_matrix[i][j] = tf_matrix[i][j] * idf_matrix[i][j]
        return tf_idf_matrix

    def cosine_similarity(self, sent1,sent2):
        s1_dot_s2=np.sum(np.multiply(sent1,sent2))
        magnitude_of_s1=math.sqrt(np.sum(np.multiply(sent1,sent1)))
        magnitude_of_s2=math.sqrt(np.sum(np.multiply(sent2,sent2)))
        return s1_dot_s2/(magnitude_of_s1*magnitude_of_s2)
    
    def cosine_matrix(self):
        cosine_similarity_matrix= np.zeros(shape=(len(self.ids_to_max_counts),len(self.ids_to_max_counts)))
        for i in range(len(self.ids_to_max_counts)):
            for j in range(len(self.ids_to_max_counts)):
                cosine_similarity_matrix[i][j] = self.cosine_similarity(self.words_to_doc_relevance[i],self.words_to_doc_relevance[j])
        for idx in range(len(cosine_similarity_matrix)):
            cosine_similarity_matrix[idx] /= cosine_similarity_matrix[idx].sum()
            
        return cosine_similarity_matrix


    def result(self):
        nx_graph = nx.from_numpy_array(self.cosine_matrix())
        scores = nx.pagerank(nx_graph)  
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(self.sent_list)), reverse=True)
        
        # Generate summary
        summary = ""
        for i in range(self.shorten_num):
            summary += f"{ranked_sentences[i][1]}. "
        return summary

# text = """Being healthy and fit in simple terms means taking good care of the body. We should remember that a healthy mind resides only in a healthy body. Good health of both mind and body helps one maintain the required energy level to achieve success in life. All of us must strive to achieve wholesome health.

# Protecting your body from the intake of harmful substances, doing regular exercises, having proper food and sleep are some of the important instances that define a healthy lifestyle. Being fit allows us to perform our activities without being lethargic, restless or tired.

# A healthy and fit person is capable of living the life to the fullest, without any major medical or physical issues. Being healthy is not only related to the physical well-being of a person, it also involves the mental stability or the internal peace of a person.

# Generally, a healthy diet consists of taking a proper and healthy food which includes eating green and fresh vegetables, fruits, having milk, eggs, minerals, proteins and vitamins essential for a human’s lifestyle. Practicing Yoga including regular exercises in your daily routine also help you maintain your desired fitness, blood sugar and immunity level.

# Healthy habits improve your physical appearance, mental stability, ability to perform activities in a better way, which help you lead a stress-free lifestyle, maintaining happy moods, high energy levels, etc. Each individual should take of one’s health on a priority; no single day should be skipped for making efforts on maintaining physical and mental fitness. Being happy is directly related to boosting your mental strength and health, so happiness can be considered as the result as well as the part of a healthy and fit lifestyle.

# """
# index = Summarizer(text)
# print(index.result())