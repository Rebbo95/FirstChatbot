import json

import nltk
import numpy as np
import pickle
import random
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = TFGPT2LMHeadModel.from_pretrained('gpt2')


class IntentClassifier:
    def __init__(self, model_file, words_file, classes_file):
        self.model_file = model_file
        self.words_file = words_file
        self.classes_file = classes_file
        self.words = []
        self.classes = []
        self.model = None

    def load_model(self):
        self.words = pickle.load(open(self.words_file, 'rb'))
        self.classes = pickle.load(open(self.classes_file, 'rb'))
        self.model = tf.keras.models.load_model(self.model_file)

    def predict_intent(self, message):
        input_data = nltk.word_tokenize(message.lower())
        input_data = [lemmatizer.lemmatize(word) for word in input_data]
        bag = [0] * len(self.words)
        for word in input_data:
            if word in self.words:
                bag[self.words.index(word)] = 1

        result = self.model.predict(np.array([bag]))[0]
        intent_index = np.argmax(result)
        intent = self.classes[intent_index]
        probability = result[intent_index]

        return [{'intent': intent, 'probability': probability}]


class ChatBot:
    def __init__(self, intent_classifier, gpt2_model):
        self.intent_classifier = intent_classifier
        self.gpt2_model = gpt2_model

    def get_response(self, message):
        intents = self.intent_classifier.predict_intent(message)
        intent = intents[0]['intent']
        if intent == '1':
            response = random.choice(intents[0]['responses'])
        else:
            response = self.gpt2_model.generate_response(message)

        return response


class GPT2Model:
    def __init__(self, config_file):
        # Add padding token to the tokenizer
        gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Load the GPT-2 model configuration from the file
        model_config = GPT2Config.from_json_file(config_file)

        # Set pad_token_id for open-end generation
        model_config.pad_token_id = model_config.eos_token_id

        # Load the GPT-2 model with the modified configuration
        self.gpt2_model = TFGPT2LMHeadModel.from_pretrained('gpt2', config=model_config)

    def generate_response(self, message):
        # Create a copy of the model's configuration
        config_copy = self.gpt2_model.config.__class__.from_dict(self.gpt2_model.config.to_dict())

        # Set pad_token_id for the copied configuration to handle padding
        config_copy.pad_token_id = gpt2_tokenizer.pad_token_id

        input_ids = gpt2_tokenizer.encode(message, return_tensors='tf')
        input_length = tf.shape(input_ids)[1]

        # Define the maximum length for the generated response
        max_response_length = input_length + 50  # You can adjust the number 50 as needed

        # Use the copied configuration in the generate method
        generated_output = self.gpt2_model.generate(
            input_ids,
            max_length=max_response_length,
            num_return_sequences=1,
            do_sample=config_copy.do_sample,
            top_p=config_copy.top_p,
            temperature=config_copy.temperature,
            pad_token_id=config_copy.pad_token_id,
            eos_token_id=config_copy.eos_token_id
        )

        generated_text = gpt2_tokenizer.decode(generated_output[0], skip_special_tokens=True)

        return generated_text


# Initialize GPT2Model first
gpt2_model = GPT2Model('config.json')

# Now create IntentClassifier and ChatBot instances
Actionbot = IntentClassifier('Models/ActionBot/ActionModel.h5', 'Models/ActionBot/words.pkl',
                             'Models/ActionBot/classes.pkl')

NewsBot = IntentClassifier('Models/NewsBot/NewsBotModel.h5', 'Models/NewsBot/words.pkl','Models/NewsBot/classes.pkl')

ActionChat = ChatBot(Actionbot, gpt2_model)

NewsChat = ChatBot(NewsBot, gpt2_model)

print("Choose your chat bot")
print("1. Action_Chatbot")
print("2. News_Chatbot")

choice = int(input("Enter your choice: "))

while True:
    if choice == 1:
        Actionbot.load_model()
        message = input("Me: ")
        response = ActionChat.get_response(message)  #Loads the ActionBot Files
        print("ActionChat:", response)

    elif choice == 2:
        NewsBot.load_model()
        message = input("Me: ")
        response = NewsChat.get_response(message)  #Loads the Newsbot Files
        print("NewsChat:", response)

    elif choice == 0:
        print("Exiting...")
        break

    else:
        print("Invalid choice. Please try again.")