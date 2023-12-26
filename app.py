import streamlit as st
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from keras.models import load_model

# タイトル
st.title("ゴルフウェア判定アプリ")

# キャプション
st.caption("このアプリケーションは、自身の服装がゴルフ場において適切かの判断に迷った際、AIがサポートしてくれるアプリケーションです。判定したい服装を着た状態で、撮影した写真（全身が望ましいですが、上半身だけでも判定は可能です。）をアップロードしてください。")

# 画像を均一なサイズにリサイズする関数
def resize_image(image_path, target_size):
    image = Image.open(image_path)
    resized_image = ImageOps.fit(image, target_size, method=0, bleed=0.0, centering=(0.5, 0.5))
    return resized_image


# 適した例と適さない例の画像
good_example_image = "img/good_example.png"
bad_example_image = "img/bad_example.png"
bad_example_image2 = "img/bad_example_2.png"

# 画像を均一なサイズにリサイズ
target_size = (300, 500)
resized_good_example = resize_image(good_example_image, target_size)
resized_bad_example1 = resize_image(bad_example_image, target_size)
resized_bad_example2 = resize_image(bad_example_image2, target_size)

# 画像を横に並べて表示
col1, col2, col3 = st.columns(3)
with col1:
    st.image(resized_good_example, caption="判定に適した画像です。", use_column_width=True)
with col2:
    st.image(resized_bad_example1, caption="人以外の画像は判定に不適です。", use_column_width=True)
with col3:
    st.image(resized_bad_example2, caption="ゴルフのポーズをとると高確率でゴルフウェアと認識されます。", use_column_width=True)


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

uploaded_file = st.file_uploader("判定したい画像をアップロードしてください。", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the selected image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Make API request to FastAPI
    url = "https://golf-fast-api.onrender.com/predict"
    
    # Use the original file name from the uploaded file
    files = {"file": (uploaded_file.name, uploaded_file.read(), "image/jpeg")}
    
    response = requests.post(url, files=files)

    # Display the prediction result
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction Result: {result['result']} (Confidence: {result['confidence']}%)")
    else:
        st.error(f"API request failed with status code {response.status_code}")
