# Sentiment Analysis Django Application

## Overview

This Django application analyzes user-provided text to classify it as either positive or negative using a sentiment analysis model based on Long Short-Term Memory (LSTM) neural network. The model is trained on the IMDb movie reviews dataset.

## Deployment

The application is deployed on the Render platform, and you can access it at [https://analyzer-6fx2.onrender.com/](https://analyzer-6fx2.onrender.com/).

## Getting Started

To use the sentiment analysis functionality locally, follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/ibrahim99035/ReviewAnalyzer.git
   cd ReviewAnalyzer
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Django development server:

   ```bash
   python manage.py runserver
   ```

4. Access the application at [http://localhost:8000/](http://localhost:8000/) in your web browser.

## Model Training and Saving

The sentiment analysis model is trained using Keras and TensorFlow. The training code can be found in the `train_model.py` file. To train and save the model, run the following command:

```bash
python train_model.py
```

The trained model will be saved in the `Model/sentiment_analysis_model.h5` file.

## Model Loading and Prediction

To load the saved model and make predictions, the `LSTM_Process` class in `lstm_process.py` is used. This class includes a `make_prediction` method that takes a text statement as input and returns the predicted sentiment (positive or negative).

```python
from lstm_process import LSTM_Process

lstm_processor = LSTM_Process()
prediction = lstm_processor.make_prediction("Your text statement goes here.")
print(prediction)
```

## Django Views

The main functionality of the application is implemented in the `views.py` file. The `home` view uses the `LSTM_Process` class to make sentiment predictions based on user input.

## Contributing

Feel free to contribute to the development of this sentiment analysis application. Fork the repository, create a new branch for your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
