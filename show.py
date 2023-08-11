import os
from flask import Flask, request,render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests, json
import csv

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    api_key = "61c578f25121eb683560eb3a9256d7b3"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    city_name = request.form['District']
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        current_temperature = y["temp"]
        current_pressure = y["pressure"]
        current_humidity = y["humidity"]
    temperature=current_temperature-273
    df = pd.read_csv("data/crop dataset.csv")
    df1=pd.read_csv("data/soil ph.csv")
    df2=pd.read_csv("data/rainfallp.csv")
    soiltype=request.form['Soiltype']
    district=str(request.form['District']).upper()
    month=str(request.form['Month']).capitalize()
    with open("data/rainfallp.csv") as f:
        reader = csv.reader(f)
        header = next(reader) # get the header row
        month_index = header.index(month) # get the index of the month
        for row in reader:
            if row[0] == district:
                rainfall = row[month_index]
                break
    ph=df1.iloc[int(soiltype)-1,1]
    le = LabelEncoder()
    df['Soiltype_encoded'] = le.fit_transform(df['Soil type'])
    features = df[['Soiltype_encoded', 'temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    sample_input = [[soiltype, temperature, current_humidity, ph, rainfall]]
    predicted_label = model.predict(sample_input)
    print("Predicted label:", predicted_label[0])
    df3 = pd.read_csv("data/samp.csv")
    filtered_data = df3[df3['commodity'] == str(predicted_label[0]).capitalize()]
    sorted_data = filtered_data.sort_values(by='modal_price', ascending=False).head(10)
    return render_template('template.html', data=sorted_data,filename=predicted_label[0])


if __name__ == '__main__':
    app.run()
