import io
import requests
import fashion_mnist_dense as model
import streamlit as st
import numpy as np
from PIL import Image

#создаем функцию загрузки файла
def load_image():
    uploaded_file = st.file_uploader(label='Âûáåðèòå èçîáðàæåíèå äëÿ ðàñïîçíàâàíèÿ')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def preprocess_image(img):
    img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


st.title('Распознавание одежды на фото)
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)
