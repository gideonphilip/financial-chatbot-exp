########################################
import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from datetime import datetime
import streamlit as st


#############################################

class BaseModel1:
    @staticmethod
    def preprocessing(data_path):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        lemma = WordNetLemmatizer()
        words = []
        classes = []
        documents = []
        ignore_words = ['?', '!']  # Ignoring symbols
        data_file = open(data_path).read()  # opening data in reading format
        intents = json.loads(data_file)  # loading json file

        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # tokenize each word
                Tokenize_word = nltk.word_tokenize(pattern)
                words.extend(Tokenize_word)
                # add document in the corpus
                documents.append((Tokenize_word, intent['tag']))

                # add to our classes list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = [lemma.lemmatize(Tokenize_word.lower()) for Tokenize_word in words if
                 Tokenize_word not in ignore_words]
        words = sorted(list(set(words)))
        # sort classes
        classes = sorted(list(set(classes)))

        pickle.dump(words, open(os.path.join(os.getcwd(), 'pickle_files', 'words.pkl'), 'wb'))
        pickle.dump(classes, open(os.path.join(os.getcwd(), 'pickle_files', 'classes.pkl'), 'wb'))

        # create our training data
        training = []
        # create an empyt array for output
        output_empty = [0] * len(classes)
        # training set , bag of words of each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # lemma each word - create base word , in attempt to represent related word
            pattern_word = [lemma.lemmatize(word.lower()) for word in pattern_words]
            # create our bag of words array with 1, if word match found in current pattern
            for Tokenize_word in words:
                bag.append(1) if Tokenize_word in pattern_words else bag.append(0)

            # output 0 and 1 in current tag(for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        # training = np.array(training)
        # create train and test list. X - pattern , y - intercept
        train_x = list(training[i][0] for i in range(len(training)))
        train_y = list(training[i][1] for i in range(len(training)))
        return train_x, train_y

    @staticmethod
    def load_model(train_x, train_y, model_name):
        model = Sequential()
        model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(train_y[0]), activation='softmax'))
        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this
        # model
        sgd = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        with st.spinner('Model Training...'):
            history = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
        st.write('Training Completed Successfully!! :)')
        model_location = os.path.join(os.getcwd(), "saved_models", model_name)
        model.save(f'{model_location}.h5')

    def train(self, data_path, model_name):
        train_x, train_y = self.preprocessing(data_path)
        self.load_model(train_x, train_y, model_name)
