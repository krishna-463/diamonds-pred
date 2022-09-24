import streamlit as st
import numpy as np
import pandas as pd
from pickle import load
from PIL import Image
import base64


st.set_page_config(page_title="Diamonds", page_icon="💎")

##################################################


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('bg.png')

##########################################

st.markdown("# Find the Diamond Price")


st.markdown("### Which type of cut do you want?")
cut = st.selectbox("CUT",['Fair','Good','Very Good','Ideal','Premium'])
cut_encoder = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Ideal' : 4, 'Premium' : 5}
cut_label_encoded = cut_encoder[cut]


st.markdown("### From below image select type of colour do you want?")
pic1 = Image.open('color.png')
st.image(pic1)
clr = st.selectbox('COLOR',['J', 'I', 'H', 'G','F', 'E','D'])
color_encoder = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}
color_label_encoded = color_encoder[clr]


st.markdown("### Which type of clarity do you want?")
pic2 = Image.open('clarity.png')
st.image(pic2)
clarity = st.selectbox('CLARITY',['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
clarity_encoder = {'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8}
clarity_label_encoded = clarity_encoder[clarity]


qp_cat = np.array([cut_label_encoded,color_label_encoded,clarity_label_encoded])

st.markdown("### Carat value do you want?")
pic3 = Image.open('carat.png')
st.image(pic3)
carat = st.slider('CARAT', 0.0, 10.0,step = 0.01 )


st.markdown("### Value of X")
x = st.slider('X', 0.0, 100.0,step = 0.001 )


st.markdown("### Value of Y")
y = st.slider('Y', 0.0, 100.0,step = 0.001 )


st.markdown("### Value of Z")
z = st.slider('Z', 0.0, 100.0,step = 0.001 )


st.markdown("### Value of Depth")
depth = st.slider('DEPTH', 30.0, 100.0,step = 0.01 )


st.markdown("### Value of Table")
table = st.slider('TABLE', 30.0, 100.0,step = 0.01 )


btn_click = st.button('FIND')


scaling = load(open('models/scaling.pkl','rb'))
model = load(open('models/dt.pkl','rb'))

if btn_click == True:
    qp = np.array([float(carat),float(depth),float(table),float(x),float(y),float(z)])
    qp.reshape(1,-1)
    qp_scaled = scaling.transform(qp.reshape(1,-1))
    qp_transformed = np.append(qp_scaled,qp_cat)
    answer = model.predict(qp_transformed.reshape(1,-1))
    t = "Price of the Diamond is:$" +str(answer[0])
    st.header(t)



