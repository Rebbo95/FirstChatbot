import json
import nltk
import numpy as np
import pickle
import random
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = TFGPT2LMHeadModel.from_pretrained('gpt2')


def generate_response(message):
    encoded_input = gpt2_tokenizer.encode(message, return_tensors='tf')
    generated_output = gpt2_model.generate(encoded_input, max_length=50, num_return_sequences=1)
    generated_text = gpt2_tokenizer.decode(generated_output[0], skip_special_tokens=True)

    return generated_text


class Trainer:
    def __init__(self, intent_file, model_file):
        self.intent_file = intent_file
        self.model_file = model_file

    def load_data(self):
        with open(self.intent_file) as file:
            data = json.load(file)

        intents = data['intents']
        words = []
        classes = []
        documents = []
        ignore_letters = ['?', '!']

        for intent in intents:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                words.extend(word_list)
                documents.append((word_list, intent['tag']))

                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
        words = sorted(set(words))

        classes = sorted(set(classes))

        training = []
        output_empty = [0] * len(classes)

        for document in documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        return words, classes, train_x, train_y

    def create_model(self, input_shape, output_shape):
        model = Sequential()
        model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_shape, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train_model(self, train_x, train_y, model):
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        model.save(self.model_file, hist)

    def get_response(self, message):
        response = generate_response(message)
        return response

    def train(self):
        words, classes, train_x, train_y = self.load_data()
        model = self.create_model(len(train_x[0]), len(train_y[0]))
        self.train_model(train_x, train_y, model)

        pickle.dump(words, open('../Models/NewsBot/words.pkl','wb'))
        pickle.dump(classes, open('../Models/NewsBot/classes.pkl', 'wb'))

        print('Done')


intent_file = '../api_data.json'
model_file = '../Models/NewsBot/NewsBotModel.h5'

trainer = Trainer(intent_file, model_file)
trainer.train()
