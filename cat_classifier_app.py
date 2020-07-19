# Make sure you complete the following commands before moving to the next steps:
# git clone https://github.com/ultralytics/google-images-download
# cd google-images-download
# pip install -U -r requirements.txt
# python3 bing_scraper.py --url 'https://www.bing.com/images/search?q=domestic+short+hair'
# --limit 100 --download --chromedriver /usr/local/bin/chromedriver
# python3 bing_scraper.py --url 'https://www.bing.com/images/search?q=british+short+hair'
# --limit 100 --download --chromedriver /usr/local/bin/chromedriver
# python3 bing_scraper.py --url 'https://www.bing.com/images/search?q=korat+cat'
# --limit 100 --download --chromedriver /usr/local/bin/chromedriver
# python3 bing_scraper.py --url 'https://www.bing.com/images/search?q=persian+cat'
# --limit 100 --download --chromedriver /usr/local/bin/chromedriver

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st


PATH = "/Users/camngn/Downloads/Coding_projects/PyCharm_Projects/TensorFlow/"
WEIGHTS = "cat_classifier_model.h5"
CLASS_DICT = {
    0: 'domestic short hair',
    1: 'British short hair',
    2: 'korat',
    3: 'Persian'
}


def load_img(input_image, shape):
    img = Image.open(input_image).convert('RGB')
    img = img.resize((shape, shape))
    img = image.img_to_array(img)
    return np.reshape(img, [1, shape, shape, 3])/255


@st.cache(allow_output_mutation=True)
def load_own_model(weights):
    return load_model(weights)


if __name__ == "__main__":
    result = st.empty()
    uploaded_img = st.file_uploader(label='Upload your image:')
    if uploaded_img:
        st.image(uploaded_img, caption="This is your cat photo.",
                 width=350)
        result.info("please wait for your results")
        model = load_own_model(PATH + WEIGHTS)
        pred_img = load_img(uploaded_img, 224)
        pred = CLASS_DICT[np.argmax(model.predict(pred_img))]
        result.success("The breed of cat is " + pred + '.')
