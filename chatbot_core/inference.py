##########################################################
import re
import pandas as pd
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
import nltk
import json
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


#############################################################

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    lemma = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word -create short form for word
    sentence_words = [lemma.lemmatize(
        word.lower()) for word in sentence_words]
    # print(sentence_words)
    return sentence_words
    # return bag of words array: 0 and 1 for each word in the bag that exits in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                # print(bag)
                if show_details:
                    print("found in bag: %s" % w)
    # print(np.array(bag))
    return (np.array(bag))


def predict_class(sentence, words, classes, model):
    # filter out prediction below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    # print(res)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # print(return_list)
    return return_list


def custom_cosine_similarity(user_input, response):
    documents = [user_input, response]
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names_out(),
                      index=['user_input', 'response'])
    return cosine_similarity(df, df)[0][1]


def getResponse(input_text, ints, intents_json):
    tag = ints[0]['intent']
    # print(tag)
    list_of_intents = intents_json['intents']
    # print(list_of_intents)
    for index in list_of_intents:

        if index['tag'] == tag:

            if index['tag'] in ['greetings', 'goodbye', 'thanks']:
                result = random.choice(index['responses'])
                return result
            elif index['tag'] is ['connect']:
                result = index['response']

            else:
                similarity_score = []
                for i, value in enumerate(index['responses']):
                    score = custom_cosine_similarity(input_text, value)
                    if score > 0:
                        similarity_score.append([score, i])
                    else:
                        continue
                if len(similarity_score) == 0:
                    result = "Please, ask your question in detail"
                elif len(similarity_score) == 1:
                    result = index['responses'][similarity_score[0][1]]
                else:
                    for value in range(len(similarity_score)):
                        if similarity_score[value][0] == np.sort(similarity_score, axis=0)[-1][0]:
                            index_value = similarity_score[value][1]
                            result = index['responses'][index_value]
                            break

                return result


def chatbot_response(text, data_path, model):
    intents = json.loads(open(data_path).read())
    words = pickle.load(open(os.path.join(os.getcwd(), 'pickle_files', 'words.pkl'), 'rb'))
    classes = pickle.load(open(os.path.join(os.getcwd(), 'pickle_files', 'classes.pkl'), 'rb'))
    ints = predict_class(text, words, classes, model)
    res = getResponse(text, ints, intents)
    return res
