import streamlit as stm
stm.set_page_config(
    page_title="ImageCaptionGenrator",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
from PIL import Image
import argparse
import numpy as np
import string
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import os
from pickle import dump, load

# Tensorflow Imports
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merging import add
from keras.models import Model, load_model



# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help="Image Path")
# args = vars(ap.parse_args())
# img_path = args['image']


def save_upload_img(upload_img):
    try:
        # create_directory(dirs= [upload_path])
        with open(os.path.join('artifacts/upload', upload_img.name), 'wb') as f:
            f.write(upload_img.getbuffer())
        
        return True 
    
    except:
        return False
    

def extract_features(model, filename):
        try:
            image = Image.open(filename)

        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = Image.open(filename)
        image= image
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
         if index == integer:
                return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


img_path = 'Flickr8k_Dataset/Flicker8k_Dataset/1001773457_577c3a7d70.jpg'
max_length = 32
tokenizer = load(open("tokens.pkl","rb"))
model = load_model('model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

# photo = extract_features(xception_model, img_path)
# img = Image.open(img_path)


# print("\n\n")
# print(description)
# plt.imshow(img)    
    
upload_path= 'artifacts/upload'
stm.title('Image Caption Generator')
stm.write('Uses CNN-RNN Model to Predict Caption of Image. Check out the GitHub Repo',"[here](https://github.com/Dev228-afk/Image-Caption-Generator)")
stm.write("Connect with me on : ","[![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://gitHub.com/Dev228-afk)", 
"\t[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dev-ansodariya-b616941b9)",
"\t[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/devansodariya)")

upload_img= stm.file_uploader('choose an image', type=['jpg','png','jpeg'])
if upload_img is not None:
    # save img
        #try:
            if save_upload_img(upload_img):
                # load img
                display_img= Image.open(upload_img)
            with stm.spinner('\tWait for it...'):
                # extract features
                features= extract_features(xception_model , os.path.join(upload_path, upload_img.name) )
               
                # suggestion
                description = generate_desc(model, tokenizer, features, max_length)[5:-3]
                
            stm.success('Done!')    
            col1, col2= stm.columns(2)
            with col1:
                 stm.image(display_img,width=350)
            with col2:
                 stm.header('Seems Like ' + description)
                
        