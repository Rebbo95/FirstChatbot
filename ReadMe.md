# ChatBot By Robin

## Introduction
This Markdown file provides an overview of a ChatBot implementation that incorporates an Intent Classifier and the GPT-2 language model. The ChatBot is capable of handling two different conversational tasks: Action ChatBot and News ChatBot. It uses an Intent Classifier to understand the user's intent and then generates appropriate responses using the GPT-2 language model.

## Installation
Before running the ChatBot, you need to install the required dependencies. The following packages are necessary for the code to function correctly:

- `json`: This is a standard Python library for working with JSON data.
- `nltk`: Natural Language Toolkit, used for tokenization and word lemmatization.
- `numpy`: Required for numerical operations.
- `pickle`: Used for reading and writing Python objects in a serialized format.
- `random`: Necessary for generating random responses.
- `tensorflow`: Deep learning framework for building and training models.
- `transformers`: Hugging Face's Transformers library, used for GPT-2 model loading and tokenization.
- `Added a batfile`, that should install the required dependencies.
## Implementation
The code provided in this Markdown file consists of five main sections:

### 1. Importing Required Libraries
The first section of the code includes importing the necessary Python libraries for the ChatBot implementation. These libraries are `json`, `nltk`, `numpy`, `pickle`, `random`, `tensorflow`, and `transformers`.

### 2. Intent Classifier
The second section defines the `IntentClassifier` class. This class is responsible for loading the trained model and predicting the user's intent based on the input message. The intent classifier uses `nltk` for tokenization and lemmatization, and a trained TensorFlow model for classification.

### 3. GPT-2 Language Model
The third section defines the `GPT2Model` class. This class loads the GPT-2 language model using the `transformers` library. It also handles generating a response based on the user's message.

### 4. ChatBot Class
The fourth section implements the `ChatBot` class, which brings together the Intent Classifier and the GPT-2 Model. It takes an instance of the Intent Classifier and the GPT-2 Model as input. The ChatBot class uses the Intent Classifier to understand the user's intent and decides whether to use the ActionBot or the NewsBot.

### 5. Chatting with the Bot
The final section of the code allows users to interact with the ChatBot. It first initializes the GPT-2 Model and creates instances of the ActionBot and NewsBot using the Intent Classifier. The user is then prompted to choose the ChatBot they want to interact with. The ChatBot loads the relevant model based on the user's choice and starts the conversation. The user can type messages, and the ChatBot responds accordingly.

Please note that the models used in this implementation, i.e., ActionBot and NewsBot, currently contain small amounts of trained data for demonstration purposes. The files required for each model, such as `classes.pkl`, `words.pkl`, and the model file (`model.h5`), are expected to be available in the specified locations. For a fully functional ChatBot, you would need to train the Intent Classifier on a more comprehensive dataset.

To run the ChatBot, simply execute the Python code provided. It will prompt you to choose between interacting with the Action ChatBot or the News ChatBot. Enter your choice, and you can start chatting with the selected ChatBot.

Enjoy!

## to do:
- fix the length of the generated response
- Train more data
-  Add more "chatbots"
- probably something, else, still learning :)
