from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .interface import LSTM_Process

@csrf_exempt
def home(request):
    LSTM_Object = LSTM_Process()
    prediction = ''
    if request.method == 'POST':
        input_text = request.POST.get('statementInput', '')
        prediction = LSTM_Object.make_prediction(input_text)
        
    return render(request, 'home.html', {'prediction': prediction})