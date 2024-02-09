import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

CLASS_NAMES = ['scottish_deerhound','maltese_dog','bernese_mountain_dog']

def predictDog(image):
    model = tf.keras.models.load_model("./dogBreedPredicition.h5")
    img = image.resize((224, 224))
    img = np.array(img, dtype='float32')
    x = np.expand_dims(img.copy(), axis=0)
    I = x/255

    print(I.shape)
    plt.imshow(I[0])
    print("model.summary")
    print(model.summary())
    img = I.reshape((1,224,224,3))
    pred= model.predict(img)
    print(pred)
    result = np.argmax(pred)
    return CLASS_NAMES[result]

# Steamlit 

st.title('Dog Breed Prediction')
uploaded_file = st.file_uploader("choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    result= predictDog(image)
    st.image(image)
    st.write("The breed is ",result)