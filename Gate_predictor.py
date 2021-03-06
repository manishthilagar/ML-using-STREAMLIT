#importing packages

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


from PIL import Image
img=Image.open("02.jpg")
st.image(img)


st.title("Gate Rank Predictor")
user_input = st.sidebar.text_input("Enter Your Name")


X0 = st.sidebar.slider('Enter Number of Hours', 1, 15)
y0 = st.sidebar.slider('Enter Months of Preparation', 1, 20)



#importing dataframe
df = pd.read_csv("gate.csv")
X = df[['hour','months']]
y = df['rank']


#creating model
clf = LinearRegression().fit(X, y)

#model fitting
clf.fit(X,y)

#making prediction
predict = clf.predict([(X0,y0)])
predict = int(predict)

#printing rank
if st.sidebar.button("Get Rank"):
    st.write("###",user_input,"your rank will be:",predict)
    st.info("Your are Awesome !")
