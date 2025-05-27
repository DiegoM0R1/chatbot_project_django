# chatbot_app/model_trainer.py
import json
import os
import numpy as np
import pickle
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Assuming nlp_utils.py is in the same directory (chatbot_app)
from .nlp_utils import limpiar_texto # Use relative import

# --- Configuration ---
# Get the base directory of the Django project
# model_trainer.py is in chatbot_app, so BASE_DIR is two levels up.
# If manage.py is at project root, this should point to project root.
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = CURRENT_SCRIPT_DIR # chatbot_app directory
PROJECT_ROOT_DIR = os.path.dirname(APP_DIR) # This should be where manage.py is

DATA_FILE = os.path.join(APP_DIR, 'data', 'training_data.json')
MODEL_DIR_NAME = 'model' # Name of the model directory within the app
MODEL_FULL_DIR = os.path.join(APP_DIR, MODEL_DIR_NAME) # Full path to model dir

MODEL_PATH = os.path.join(MODEL_FULL_DIR, 'intent_nn_model.h5')

TOKENIZER_PATH = os.path.join(MODEL_FULL_DIR, 'tokenizer.pickle')
LABEL_ENCODER_PATH = os.path.join(MODEL_FULL_DIR, 'label_encoder.joblib')


# Hyperparameters (same as before)
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
MAX_LEN = 25
LSTM_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.5
EPOCHS = 100
BATCH_SIZE = 16
OOV_TOKEN = "<OOV>"

def cargar_datos():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['intents'] # <--- CAMBIA ESTO A 'intents'


def preparar_datos(intents_data):
    patterns = []
    tags = []

    for intent in intents_data:
        if intent['tag'] == 'desconocido' and not intent['patterns']:
            continue
        for pattern in intent['patterns']:
            cleaned_pattern = limpiar_texto(pattern)
            if cleaned_pattern:
                patterns.append(cleaned_pattern)
                tags.append(intent['tag'])
    
    if not patterns:
        raise ValueError("No training patterns found. Check training_data.json.")

    label_encoder = LabelEncoder()
    integer_encoded_tags = label_encoder.fit_transform(tags)
    one_hot_labels = to_categorical(integer_encoded_tags, num_classes=len(label_encoder.classes_))
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(patterns)
    word_index = tokenizer.word_index
    
    sequences = tokenizer.texts_to_sequences(patterns)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    return padded_sequences, one_hot_labels, tokenizer, label_encoder, len(word_index) + 1

def crear_modelo_nn(input_shape, num_classes, actual_vocab_size):
    print(f"Creando modelo con input_shape: {input_shape}, num_classes: {num_classes}, vocab_size: {actual_vocab_size}")
    model = Sequential([
        Input(shape=input_shape, name='input_layer'),
        Embedding(input_dim=actual_vocab_size, 
                  output_dim=EMBEDDING_DIM, 
                  input_length=MAX_LEN, 
                  name='embedding_layer'),
        LSTM(LSTM_UNITS, name='lstm_layer', return_sequences=False),
        Dense(DENSE_UNITS, activation='relu', name='dense_layer_1'),
        Dropout(DROPOUT_RATE, name='dropout_layer'),
        Dense(num_classes, activation='softmax', name='output_layer')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def entrenar_y_guardar_modelo():
    intents_data = cargar_datos()
    padded_sequences, one_hot_labels, tokenizer, label_encoder, actual_vocab_size = preparar_datos(intents_data)
    
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, one_hot_labels, test_size=0.2, random_state=42, stratify=one_hot_labels.argmax(axis=1)
    )
    
    num_classes = y_train.shape[1]
    model = crear_modelo_nn(input_shape=(MAX_LEN,), num_classes=num_classes, actual_vocab_size=min(VOCAB_SIZE, actual_vocab_size))
    
    print("\n--- Iniciando Entrenamiento del Modelo Neuronal (Django context) ---")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping],
                        verbose=1)
    
    print("\n--- Entrenamiento Completado ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Pérdida en Test: {loss:.4f}")
    print(f"Precisión en Test: {accuracy*100:.2f}%")
    
    # Guardar
    if not os.path.exists(MODEL_FULL_DIR):
        os.makedirs(MODEL_FULL_DIR)
        
    model.save(MODEL_PATH)
    print(f"Modelo Keras guardado en: {MODEL_PATH}")
    
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer Keras guardado en: {TOKENIZER_PATH}")
    
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"LabelEncoder guardado en: {LABEL_ENCODER_PATH}")

if __name__ == "__main__":
    # This allows running the script directly: `python chatbot_app/model_trainer.py`
    # Ensure your current working directory is the project root (`chatbot_project_django`)
    # OR that this script correctly resolves paths if run from elsewhere.
    # The paths are now defined from APP_DIR, so it should be fine if run from anywhere.
    print(f"Usando DATA_FILE: {DATA_FILE}")
    print(f"Guardando modelos en MODEL_FULL_DIR: {MODEL_FULL_DIR}")
    entrenar_y_guardar_modelo()