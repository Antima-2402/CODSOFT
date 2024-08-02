import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, add, GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image


# Load the VGG16 model
vgg = VGG16(weights='imagenet', include_top=False)
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-1].output)

def extract_features(filename, model):
    """Extract features from an image using a pre-trained model"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features

# Define the captioning model
def define_model(vocab_size, max_length):
    inputs1 = tf.keras.layers.Input(shape=(7, 7, 512))
    fe1 = GlobalAveragePooling2D()(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = tf.keras.layers.Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)
    
    decoder1 = add([fe2, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Sample dataset for captions
captions = {
    "example.JPEG.JPEG": ["startseq a man riding a horse on a beach endseq"]
}

def preprocess_captions(captions):
    all_captions = []
    for key in captions:
        all_captions.extend(captions[key])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)
    return tokenizer, vocab_size, max_length

tokenizer, vocab_size, max_length = preprocess_captions(captions)

def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Example usage
filename = 'example.JPEG.JPEG'  # Ensure this file exists in the directory or provide the correct path
photo = extract_features(filename, vgg)
photo = photo.reshape((7, 7, 512))  # Ensure correct shape
desc_list = captions[filename]
X1, X2, y = create_sequences(tokenizer, max_length, desc_list, photo)

# Reshape X1 to match expected input shape (num_samples, 7, 7, 512)
X1 = X1.reshape((X1.shape[0], 7, 7, 512))

# Debug: Print the shapes of the arrays
print("Shape of X1:", X1.shape)  # Should be (num_samples, 7, 7, 512)
print("Shape of X2:", X2.shape)  # Should be (num_samples, max_length)
print("Shape of y:", y.shape)    # Should be (num_samples, vocab_size)

# Define and compile the model
model = define_model(vocab_size, max_length)
model.summary()

# Train the model (using the sample data; in practice, use a larger dataset)
model.fit([X1, X2], y, epochs=10, verbose=2)

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo.reshape((1, 7, 7, 512)), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Generate a description for the example image
description = generate_desc(model, tokenizer, photo, max_length)
print(description)

# Display the image
image = Image.open(filename)
plt.imshow(image)
plt.axis('off')
plt.show()
