from django.apps import AppConfig
import os 
import pickle
import joblib
import json
from tensorflow.keras.models import load_model

MODEL_NN = None
TOKENIZER = None
LABEL_ENCODER = None
RESPONSES_DICT = None
SUGGESTIONS_DICT = None
FULL_TRAINING_DATA = None 

MAX_LEN_MODEL = 25
CONFIDENCE_THRESHOLD_MODEL = 0.45
TAG_DESCONOCIDO = "desconocido"

class ChatbotAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot_app'

    def ready(self):
        global MODEL_NN, TOKENIZER, LABEL_ENCODER, RESPONSES_DICT, SUGGESTIONS_DICT, FULL_TRAINING_DATA, TAG_DESCONOCIDO 

        print("--- DEBUG: ChatbotAppConfig.ready() called ---")

        app_path = os.path.dirname(__file__)
        model_dir = os.path.join(app_path, 'model')
        data_dir = os.path.join(app_path, 'data')

        model_path_keras = os.path.join(model_dir, 'intent_nn_model.h5')
        tokenizer_path_pickle = os.path.join(model_dir, 'tokenizer.pickle')
        label_encoder_path_joblib = os.path.join(model_dir, 'label_encoder.joblib')
        data_file_json = os.path.join(data_dir, 'training_data.json')

        print(f"DEBUG: Base app_path: {app_path}")
        print(f"DEBUG: Model path: {model_path_keras}")
        print(f"DEBUG: Tokenizer path: {tokenizer_path_pickle}")
        print(f"DEBUG: Label Encoder path: {label_encoder_path_joblib}")
        print(f"DEBUG: Data JSON path: {data_file_json}")

        try:
            keras_exists = os.path.exists(model_path_keras)
            tokenizer_exists = os.path.exists(tokenizer_path_pickle)
            label_encoder_exists = os.path.exists(label_encoder_path_joblib)
            data_file_exists = os.path.exists(data_file_json)

            if keras_exists and tokenizer_exists and label_encoder_exists and data_file_exists:
                print("DEBUG: All critical files reported as existing. Attempting to load...")
                MODEL_NN = load_model(model_path_keras)
                
                with open(tokenizer_path_pickle, 'rb') as handle:
                    TOKENIZER = pickle.load(handle)
                
                LABEL_ENCODER = joblib.load(label_encoder_path_joblib)
                
                with open(data_file_json, 'r', encoding='utf-8') as f:
                    FULL_TRAINING_DATA = json.load(f) 
                
                if FULL_TRAINING_DATA: 
                    intents_data_list = FULL_TRAINING_DATA.get('intents', [])
                    RESPONSES_DICT = {intent['tag']: intent.get('responses', ["(Respuesta no definida para este tag)"]) for intent in intents_data_list}
                    SUGGESTIONS_DICT = {intent['tag']: intent.get('suggestions', []) for intent in intents_data_list}
                else:
                    print("ERROR ChatbotAppConfig: FULL_TRAINING_DATA was None or empty after loading JSON (training_data.json might be empty or corrupt).")
                    FULL_TRAINING_DATA = None 
                    RESPONSES_DICT = {TAG_DESCONOCIDO: ["Error al cargar datos de configuración (JSON vacío o corrupto)."]}
                    SUGGESTIONS_DICT = {TAG_DESCONOCIDO: []}

                # Ensure defaults for TAG_DESCONOCIDO if not present in the loaded intents
                if RESPONSES_DICT is not None and TAG_DESCONOCIDO not in RESPONSES_DICT:
                    RESPONSES_DICT[TAG_DESCONOCIDO] = ["Lo siento, no pude entenderte bien en este momento."]
                if SUGGESTIONS_DICT is not None and TAG_DESCONOCIDO not in SUGGESTIONS_DICT:
                    SUGGESTIONS_DICT[TAG_DESCONOCIDO] = []
                
                print("Chatbot Keras models, tokenizer, label encoder, full training data, responses, and suggestions loaded successfully.")
            
            else: 
                missing_files_debug = []
                if not keras_exists: missing_files_debug.append(f"Keras model (path: {model_path_keras})")
                if not tokenizer_exists: missing_files_debug.append(f"Tokenizer (path: {tokenizer_path_pickle})")
                if not label_encoder_exists: missing_files_debug.append(f"Label encoder (path: {label_encoder_path_joblib})")
                if not data_file_exists: missing_files_debug.append(f"Data file (path: {data_file_json})")
                
                print(f"Chatbot WARNING: One or more critical files are missing: {', '.join(missing_files_debug)}.")
                print("Chatbot will not function correctly. Setting core components to None.")
                MODEL_NN = None
                TOKENIZER = None
                LABEL_ENCODER = None
                FULL_TRAINING_DATA = None 
                RESPONSES_DICT = {TAG_DESCONOCIDO: ["Error: Faltan archivos esenciales del chatbot."]}
                SUGGESTIONS_DICT = {TAG_DESCONOCIDO: []}
        
        except Exception as e: 
            print(f"Chatbot CRITICAL ERROR (Exception during load attempt in apps.py ready()): {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc() 
            print("Setting all chatbot core components to None due to critical exception during loading.")
            MODEL_NN = None
            TOKENIZER = None
            LABEL_ENCODER = None
            FULL_TRAINING_DATA = None 
            RESPONSES_DICT = {TAG_DESCONOCIDO: ["Error crítico del sistema chatbot (excepción en carga)."]}
            SUGGESTIONS_DICT = {TAG_DESCONOCIDO: []}
        
        print("--- DEBUG: ChatbotAppConfig.ready() finished ---")