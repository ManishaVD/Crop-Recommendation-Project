from django.shortcuts import render
from sklearn.model_selection import train_test_split
# Create your views here.
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

# Create your views here.
def crop(request):
    return render(request,'crop.html')

def about(request):
    return render(request,'about.html')

def index(request):
    return render(request,'index.html')

def contact(request):
    return render(request,'contact.html')

def advisor(request):
    return render(request, 'advisor.html')

def newadvisor(request):
    return render(request, 'newadvisor.html')

def chart(request):
    return render(request, 'chart.html')


def classify_crop(N, P, K, temp_input, humidity_input, pH_input, rain_input):
    data = pd.read_csv("C:/Users/admin/Documents/Aman sirs assignment/projectAman/mainproject/mainapp/data/Crop_recommendation (2).csv")
    data_x = data.iloc[:, :7]
    data_y = data.iloc[:, -1]
    crop_rfc_model = RandomForestClassifier(n_estimators=100)
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=101)
    crop_rfc_model.fit(data_x_train, data_y_train)
    test = np.array([[N, P, K, temp_input, humidity_input, pH_input, rain_input]])
    prediction_crop = crop_rfc_model.predict(test)

    crop_labels = {
    'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
    'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
    'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14,
    'apple': 15, 'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19,
    'jute': 20, 'coffee': 21
    }
    crop_label = {
    0: 'rice',
    1: 'maize',
    2: 'chickpea',
    3: 'kidneybeans',
    4: 'pigeonpeas',
    5: 'mothbeans',
    6: 'mungbean',
    7: 'blackgram',
    8: 'lentil',
    9: 'pomegranate',
    10: 'banana',
    11: 'mango',
    12: 'grapes',
    13: 'watermelon',
    14: 'muskmelon',
    15: 'apple',
    16: 'orange',
    17: 'papaya',
    18: 'coconut',
    19: 'cotton',
    20: 'jute',
    21: 'coffee'
    }

    if prediction_crop.item() in crop_labels:
        predicted_crop_label = crop_labels[prediction_crop.item()]
        if predicted_crop_label in crop_label:
            result = crop_label[predicted_crop_label]
        return result
    else:
        return 'Invalid choice'
# In your view
def result2(request):
    if request.method == 'POST':
        N = float(request.POST.get("nit"))
        P = float(request.POST.get("phos"))
        K = float(request.POST.get("pot"))
        temp_input = float(request.POST.get("tem"))
        humidity_input = float(request.POST.get("hum"))
        pH_input = float(request.POST.get("ph"))
        rain_input = float(request.POST.get("rainf"))

        label = classify_crop(N, P, K, temp_input, humidity_input, pH_input, rain_input)

        return render(request, 'newadvisor.html', {"result2": label})
    
    return render(request, 'newadvisor.html')


def chart(request):
    return render(request,'chart.html')

def results(request):
    data = pd.read_csv("C:/Users/admin/Documents/Aman sirs assignment/projectAman/mainproject/mainapp/data/Crop_recommendation (2).csv")
    data.rename(columns={'label': 'Crop', 'N': 'Nitrogen', 'P': 'Phosphorous', 'K': 'Potassium', 'temperature': 'temperature (°C)',
                        'humidity': 'humidity (%)', 'ph': 'pH', 'rainfall': 'rainfall (mm)'}, inplace=True)
    data.Crop.replace({'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
                       'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
                       'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14, 'apple': 15,
                       'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19, 'jute': 20, 'coffee': 21}, inplace=True)

    description = pd.pivot_table(data, index=['Crop'], aggfunc='mean')
    factors = ['Nitrogen', 'Phosphorous', 'Potassium']
    crop = int(request.POST["crop"])
    share = [description.loc[crop - 1, factor] for factor in factors]


    plt.figure(figsize=(6, 6))
    explode = (0.1, 0, 0)
    plt.pie(share, labels=factors, shadow=True, startangle=90, autopct='%.2f%%', explode=explode)
    plt.title("Nutrient Requirements")
    plt.legend(loc='best')

    # Save the chart as an in-memory image
    chart_image = BytesIO()
    plt.savefig(chart_image, format="png")
    chart_image.seek(0)

    # Embed the in-memory image in the HTML template
    chart_base64 = base64.b64encode(chart_image.read()).decode('utf-8')

    return render(request, 'chart.html', {'chart_base64': chart_base64, 'factors': factors})
