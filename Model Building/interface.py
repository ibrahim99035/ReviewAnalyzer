from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model("Model/sentiment_analysis_model.h5")

# Load the file containing statements
file_path = "TextFiles/reviews.txt"

# Preprocess the statements
max_features = 5000  # Number of words to consider as features
max_len = 100  # Maximum length of sequences

# Load the statements from the file
with open(file_path, "r") as file:
    statements = file.readlines()

# Initialize tokenizer
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(statements)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(statements)

# Pad sequences to a fixed length
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Make predictions
predictions = model.predict(padded_sequences)

# Interpret predictions
sentiments = ["positive" if prediction > 0.5 else "negative" for prediction in predictions]

# Write the output to a file
predictions = "TextFiles/predictions.txt"
with open(predictions, "w") as output_file:
    for statement, sentiment in zip(statements, sentiments):
        output_file.write("Statement: {}\n".format(statement.strip()))
        output_file.write("Predicted Sentiment: {}\n".format(sentiment))
        output_file.write("--------------------\n")
