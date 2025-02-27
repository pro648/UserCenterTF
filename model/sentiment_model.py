import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

class SentimentModel:
    def __init__(self, max_words=10000, embedding_dim=128, lstm_units=64):
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = Tokenizer(num_words=self.max_words)

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=self.embedding_dim))
        model.add(LSTM(self.lstm_units, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))  # binary classification output

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess_data(self, texts, labels):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        max_len = max([len(seq) for seq in sequences])
        padded_sequences = pad_sequences(sequences, maxlen=max_len)
        return padded_sequences, np.array(labels), max_len

    def train(self, X_train, y_train, epochs=5, batch_size=2):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return accuracy
