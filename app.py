from turtle import color
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from pickle import load
from PIL import Image
#import sklearn
st.set_page_config(page_title="Diamonds", page_icon="💎")

##########################################

st.markdown("# Find the Diamond Price")

df = pd.read_csv('diamonds.csv')

cut = st.selectbox('select cut of the Diamond',df.cut.unique())
cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}
cut_label_encoded = cut_encoder[cut]

clr = st.selectbox('select color of the Diamond',df.color.unique())
color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}
color_label_encoded = color_encoder[clr]
clarity = st.selectbox('select clarity of the Diamond',df.clarity.unique())
clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}
clarity_label_encoded = clarity_encoder[clarity]
qp_cat = np.array([cut_label_encoded,color_label_encoded,clarity_label_encoded])

scaling = load(open('models/scaling.pkl','rb'))
model = load(open('models/dt.pkl','rb'))
x = st.text_input('x',placeholder='enter the value')
y = st.text_input('y',placeholder='enter the value')
z = st.text_input('z',placeholder='enter the value')
carat = st.text_input('carat',placeholder='enter the value')
depth = st.text_input('depth',placeholder='enter the value')
table = st.text_input('table',placeholder='enter the value')


btn_click = st.button('FIND')

if btn_click == True:
    if len(x) ==0 or len(y)==0 or len(z)==0 or len(carat)==0 or len(depth)==0 or len(table)==0 :
        st.error('Enter all the values')
    else:
        qp = np.array([float(carat),float(depth),float(table),float(x),float(y),float(z)])
        qp.reshape(1,-1)
        qp_scaled = scaling.transform(qp.reshape(1,-1))
        qp_transformed = np.append(qp_scaled,qp_cat)
        answer = model.predict(qp_transformed.reshape(1,-1))
        t = "Price of the Diamond is:$" +str(answer[0])
        st.header(t)



#pic1 = Image.open('carat_.jpg')
#st.image(pic1)