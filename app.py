import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
from PIL import Image
import streamlit as st

crop_recommendation_model_path = r"C:\Users\utkar\OneDrive\Desktop\Harvestify-master\models\NBClassifier.pkl"
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None
st.title('Crop Recommender')
N = st.number_input('Ratio of Nitrogen Content in soil',min_value = 0,max_value = 140, value = 10)
P = st.number_input("Ratio of Phosphorun Content in soil",min_value = 0,max_value = 140, value = 10)
K = st.number_input('Ratio of Potassium Content in soil',min_value = 0,max_value = 140, value = 10)
ph = st.number_input('pH Value of the soil',min_value = 0,max_value = 140, value = 7)
rainfall = st.number_input('Rainfall of city in mm',min_value = 0,max_value = 140, value = 118)
city = st.text_input('Enter City')
if(city and weather_fetch(city)!= None):
    temperature, humidity = weather_fetch(city)
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]
else:
    st.text('Enter Valid City')
button = st.button("Recommend")
if(button):
    st.write('You should grow {}'.format(final_prediction))
