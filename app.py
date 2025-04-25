from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import random
from random import choice, sample

# model = pickle.load(open("C:\\Users\\Dell\\complete web development\\Credit Prediction Project\\model.pkl", "rb"))
  
  
app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/prediction',methods=['POST'])
def prediction():
    data0 = request.form['age']
    data1 = request.form['sex']
    data2 = request.form['job']
    data3 = request.form['housing']
    data4 = request.form['saving']
    data5 = request.form['checking']
    data6 = request.form['credit']
    data7 = request.form['duration']
    data8 = request.form['purpose']
    # data9 = request.form['Company Name']
    if data1 == 'male':
        data1 = 1
    elif data1 == 'female':
        data1 = 0

    if data3 == 'own':
        data3 = 1
    elif data3 == 'free':
        data3 = 0
    elif data3=='rent':
        data3 = 2

    if data5 == 'NA':
        data5 = 4
    elif data5 == 'little':
        data5 = 0
    elif data5 == 'quite rich':
        data5 = 2
    elif data5 == 'rich':
        data5 = 3
    elif data5 == 'moderate':
        data5 = 1

    if data4 == 'NA':
        data4 = 3
    elif data4 == 'little':
        data4 = 0
    elif data4 == 'quite rich':
        data4 = 4
    elif data4 == 'rich':
        data4 = 2
    elif data4 == 'moderate':
        data4 = 1

    if data8 == 'education':
        data8 = 3
    elif data8 == 'business':
        data8 = 0
    elif data8 == 'furniture/equipment':
        data8 = 4
    elif data8 == 'domestic appliances':
        data8 = 2
    elif data8 == 'car':
        data8 = 1
    elif data8=='radio/TV':
        data8 = 5
    df = pd.read_csv('C:\\Users\\Dell\\complete web development\\Credit Prediction Project\\german_credit_data.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    missing_values = df.isnull().sum()
    missing_values
    df['Saving accounts'].fillna('unknown', inplace=True)
    df['Checking account'].fillna('unknown', inplace=True)

    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    df_encoded = df.copy()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le  # Save encoders for later interpretation

    scaler = StandardScaler()
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    credit_threshold = df_encoded['Credit amount'].median()
    duration_threshold = df_encoded['Duration'].median()

    df_encoded['Risk'] = (
        (df_encoded['Credit amount'] > credit_threshold) &
        (df_encoded['Duration'] < duration_threshold)
    ).astype(int)

    model = RandomForestClassifier(random_state=42)
    X = df_encoded.drop(columns=['Risk'])
    y = df_encoded['Risk']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    data = [[data0,data1,data2,data3,data4,data5,data6,data7,data8]]

    
    input_data = pd.DataFrame(data, columns= ['Age',
        'Sex',
        'Job',
        'Housing',
        'Saving accounts',
        'Checking account',
        'Credit amount',
        'Duration',
        'Purpose'],index = ['input']
    ) 

    message = model.predict(input_data)[0]

    return render_template('after.html',a=message)


@app.route('/output02')
def output02():
    return render_template('output02.html')

@app.route('/output01')
def output01():
    return render_template('output01.html')

if __name__ == "__main__":
    app.run('0.0.0.0',port=8080,debug=True)
