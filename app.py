from flask import Flask,render_template,url_for,request
#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import warnings
import numpy as np
import os

import pandas as pd
import sqlite3
import random
import smtplib 
from email.message import EmailMessage
from datetime import datetime

import os
import numpy as np
import librosa
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, RepeatVector, Bidirectional, LSTM, GRU
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.model_selection import train_test_split
import traceback
from keras.layers import Conv1D, MaxPooling1D, RepeatVector#loading CNN1D
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   
import joblib

app = Flask(__name__)
app.secret_key = 'welcome'

dataset = pd.read_csv("UrbanSound8K.csv")
labels = np.unique(dataset['class'].ravel())


labels = labels.tolist()
labels.append("crackling_fire")
labels.append("glass_breaking")
labels = np.asarray(labels)
glass_fire = pd.read_csv("50classes.csv")

#defining global variables to store X and Y training data
X = []
Y = []


X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

#visualizing class labels count found in dataset
names, count = np.unique(Y, return_counts = True)
height = count
bars = labels

import smtplib
def sendMail(email, message):
    em = []
    em.append(email)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
        email_address = 'evotingotp4@gmail.com'
        email_password = 'xowpojqyiygprhgr'
        connection.login(email_address, email_password)
        connection.sendmail(from_addr="evotingotp4@gmail.com", to_addrs=em, msg="Subject : Mail from Event Live : "+message)

@app.route('/')
def home():
	return render_template('home.html')

def getModel():
    extension_model = Sequential()
    extension_model.add(Convolution2D(32, (1 , 1), input_shape = (128, 1, 1), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Flatten())
    extension_model.add(RepeatVector(2))
    extension_model.add(Bidirectional(GRU(32,reset_after=False)))
    extension_model.add(Dense(units = 256, activation = 'relu'))
    extension_model.add(Dense(units = 12, activation = 'softmax'))
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    extension_model.load_weights("model/extension_weights.hdf5")
    return extension_model


@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")



@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/notebook')
def notebook():
	return render_template('EventDetection.html')

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():   
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        if os.path.exists("static/test.wav"):
            os.remove("static/test.wav")
        with open('static/test.wav', mode="wb") as audio:
            audio.write(img_bytes)
        audio.close()
        extension_model = getModel()
        audio, sample_rate = librosa.load('static/test.wav', res_type='kaiser_fast', sr=16000, mono=True) 
        mels = np.mean(librosa.feature.melspectrogram(y = audio, sr=sample_rate).T,axis=0)
        temp = []
        temp.append(mels)
        temp = np.asarray(temp)
        temp = np.reshape(temp, (temp.shape[0], temp.shape[1], 1, 1))
        predict = extension_model.predict(temp)#apply extension model to identify person
        predict = np.argmax(predict)
        predict = labels[predict]
        output = "Audio Threat Detected as =====> "+predict
        #change here your email address
        #sendMail("hemanthgoddalla9@gmail.com", output)#change email
        return render_template('result.html', msg=output)

if __name__ == '__main__':
	app.run(debug=False)
