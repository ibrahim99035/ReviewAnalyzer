from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class LSTM_Process:
    # Load the saved model
    model = load_model("Model/sentiment_analysis_model.h5")
    # Preprocess the statement
    max_features = 5000  # Number of words to consider as features
    max_len = 100  # Maximum length of sequences

    def make_prediction(self, statement):
        # Initialize tokenizer
        tokenizer = Tokenizer(num_words= self.max_features)
        tokenizer.fit_on_texts([statement])

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([statement])

        # Pad sequence to a fixed length
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)

        # Make prediction
        prediction = self.model.predict(padded_sequence)

        # Interpret prediction
        sentiment = "Positive" if prediction > 0.5 else "Negative"

        return sentiment

    
