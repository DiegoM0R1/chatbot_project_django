# chatbot_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie

import json
import random
import numpy as np
import re # <--- ASEGÚRATE DE TENER ESTE IMPORT

# Intenta la importación estándar de TensorFlow primero.
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    print("ADVERTENCIA: No se pudo importar pad_sequences desde tensorflow.keras. Intentando desde keras...")
    try:
        from keras.preprocessing.sequence import pad_sequences
    except ImportError as e_keras:
        print(f"ERROR CRÍTICO: No se pudo importar pad_sequences ni de tensorflow.keras ni de keras. Error: {e_keras}")
        pad_sequences = None

# Importar modelos y datos cargados globalmente desde apps.py
from .apps import (
    MODEL_NN,
    TOKENIZER,
    LABEL_ENCODER,
    RESPONSES_DICT,
    SUGGESTIONS_DICT,
    FULL_TRAINING_DATA, # <--- ¡IMPORTANTE IMPORTAR ESTO!
    MAX_LEN_MODEL,
    CONFIDENCE_THRESHOLD_MODEL,
    TAG_DESCONOCIDO
)
# Importar utilidad de limpieza de texto
from .nlp_utils import limpiar_texto

# --------------- FUNCIONES AUXILIARES PARA PROCESAMIENTO ---------------

def extraer_entidades_limpieza(cleaned_message, entidades_nlp_config):
    extracted = {'cantidad': 1} 
    match_num_digit = re.search(r'\b(\d+)\b', cleaned_message)
    if match_num_digit:
        extracted['cantidad'] = int(match_num_digit.group(1))
    else:
        numeros_config = entidades_nlp_config.get('dimensiones', {}).get('numeros', [])
        # Convertir palabras a números (ej: {'uno': 1, 'dos': 2, ...})
        # Tu training_data.json tiene "numeros": ["uno", "dos", ... , "diez"]
        # y tambien nlp.extraccion_entidades.numeros.conversion: {"uno": 1, ... "cinco": 5}
        # Usaremos el que está en entidades_nlp.dimensiones.numeros por ahora y lo mapearemos.
        # Para una solución más completa, la conversión de palabra a número debería ser más robusta.
        palabras_a_num = {palabra: i + 1 for i, palabra in enumerate(numeros_config)}
        
        # Priorizar la conversión explícita si existe en nlp_utils (no la tenemos separada, pero podríamos)
        # o desde el training_data.json más específico
        conversion_numeros_json = entidades_nlp_config.get('dimensiones', {}).get('conversion_numeros_explicita', 
                                   FULL_TRAINING_DATA.get('procesamiento_lenguaje_natural', {}).get('extraccion_entidades', {}).get('numeros', {}).get('conversion', {}))
        
        found_word_num = False
        for palabra, valor in conversion_numeros_json.items():
             if re.search(r'\b' + re.escape(palabra) + r'\b', cleaned_message, re.IGNORECASE):
                extracted['cantidad'] = valor
                found_word_num = True
                break
        if not found_word_num: # Si no se encontró en la conversión explícita, probar con la lista
            for palabra, valor in palabras_a_num.items():
                if re.search(r'\b' + re.escape(palabra) + r'\b', cleaned_message, re.IGNORECASE):
                    extracted['cantidad'] = valor
                    break

    tipos_muebles_config = entidades_nlp_config.get('tipos_muebles', {})
    mueble_encontrado_display = None
    categoria_precio_mueble = None
    map_categoria_a_precio_base_key = {
        "pequenos": "silla",
        "medianos": "mueble_mediano",
        "grandes": "mueble_grande"
    }
    for categoria_json, lista_muebles_keyword in tipos_muebles_config.items():
        for keyword in lista_muebles_keyword:
            if re.search(r'\b' + re.escape(keyword) + r'\b', cleaned_message, re.IGNORECASE):
                mueble_encontrado_display = keyword
                categoria_precio_mueble = map_categoria_a_precio_base_key.get(categoria_json)
                break
        if mueble_encontrado_display:
            break
    if mueble_encontrado_display:
        extracted['tipo_mueble_display'] = mueble_encontrado_display
    else:
        extracted['tipo_mueble_display'] = "muebles"
    if categoria_precio_mueble:
        extracted['tipo_mueble_categoria_precio'] = categoria_precio_mueble
    else:
        extracted['tipo_mueble_categoria_precio'] = map_categoria_a_precio_base_key.get("medianos")
    return extracted

def calcular_precio_limpieza(entidades_extraidas, sistema_calculo_config):
    resultado = {'calculo_detallado': "No se pudo calcular el precio. Por favor, especifica mejor."}
    config_limpieza = sistema_calculo_config.get('funciones_precio', {}).get('calcular_limpieza', {})
    precios_base = config_limpieza.get('precios_base', {})
    cantidad = entidades_extraidas.get('cantidad', 1)
    categoria_precio = entidades_extraidas.get('tipo_mueble_categoria_precio')
    if categoria_precio and categoria_precio in precios_base:
        precio_unitario = precios_base[categoria_precio]
        precio_total = precio_unitario * cantidad
        resultado['calculo_detallado'] = f"S/ {precio_total:.2f}"
        resultado['precio_total_raw'] = precio_total
    else:
        if categoria_precio:
             resultado['calculo_detallado'] = f"No tenemos un precio base para la categoría '{categoria_precio}'. Consulta por otro tipo de mueble."
    return resultado

# --------------- FIN FUNCIONES AUXILIARES ---------------

@method_decorator(ensure_csrf_cookie, name='dispatch')
class HomeView(View):
    def get(self, request, *args, **kwargs):
        print(f"DEBUG HomeView: Renderizando index.html. Session ID: {request.session.session_key}")
        return render(request, 'chatbot_app/index.html')

@method_decorator(csrf_exempt, name='dispatch')
class ChatAPIView(View):
    def post(self, request, *args, **kwargs):
        print(f"\n--- DEBUG ChatAPIView: POST request received. Session ID: {request.session.session_key} ---")
        
        required_components = {
            "MODEL_NN": MODEL_NN, "TOKENIZER": TOKENIZER, "LABEL_ENCODER": LABEL_ENCODER,
            "RESPONSES_DICT": RESPONSES_DICT, "SUGGESTIONS_DICT": SUGGESTIONS_DICT,
            "FULL_TRAINING_DATA": FULL_TRAINING_DATA # <--- AÑADIDO A LA VERIFICACIÓN
        }
        missing_components = [name for name, comp in required_components.items() if comp is None]
        if missing_components:
            error_msg = f"ERROR ChatAPIView: Componentes faltantes: {', '.join(missing_components)}."
            print(error_msg)
            return JsonResponse({'error': 'Chatbot no disponible (error de configuración).', 'suggestions': []}, status=500)
        
        if pad_sequences is None:
            print("ERROR ChatAPIView: pad_sequences no está disponible.")
            return JsonResponse({'error': 'Error interno del servidor (pad_sequences).', 'suggestions': []}, status=500)

        try:
            try:
                data = json.loads(request.body)
                user_message = data.get('message', '')
            except json.JSONDecodeError:
                print("ERROR ChatAPIView: JSON inválido.")
                return JsonResponse({'error': 'JSON inválido.', 'suggestions': []}, status=400)

            print(f"DEBUG ChatAPIView: Mensaje original: '{user_message}'")
            cleaned_message = limpiar_texto(user_message)
            print(f"DEBUG ChatAPIView: Mensaje limpiado: '{cleaned_message}'")

            if not cleaned_message:
                bot_response = random.choice(RESPONSES_DICT.get(TAG_DESCONOCIDO, ["No entendí."]))
                bot_suggestions = SUGGESTIONS_DICT.get(TAG_DESCONOCIDO, [])
                return JsonResponse({'response': bot_response, 'suggestions': bot_suggestions})

            sequence = TOKENIZER.texts_to_sequences([cleaned_message])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN_MODEL, padding='post', truncating='post')
            prediction_probs = MODEL_NN.predict(padded_sequence, verbose=0)[0]
            predicted_index = np.argmax(prediction_probs)
            confidence = float(np.max(prediction_probs))
            print(f"DEBUG ChatAPIView: Probs: {prediction_probs.tolist()} -> Index: {predicted_index} -> Conf: {confidence:.4f}")

            predicted_tag_final = TAG_DESCONOCIDO
            candidate_tag = LABEL_ENCODER.inverse_transform([predicted_index])[0] if predicted_index < len(LABEL_ENCODER.classes_) else TAG_DESCONOCIDO

            if confidence >= CONFIDENCE_THRESHOLD_MODEL and candidate_tag in RESPONSES_DICT:
                predicted_tag_final = candidate_tag
                print(f"DEBUG ChatAPIView: Tag aceptado '{candidate_tag}'.")
            else:
                print(f"DEBUG ChatAPIView: Confianza {confidence:.4f} (Umbral {CONFIDENCE_THRESHOLD_MODEL}) o tag '{candidate_tag}' no en RESPONSES. Usando '{TAG_DESCONOCIDO}'.")
            
            bot_response_template_list = RESPONSES_DICT.get(predicted_tag_final, RESPONSES_DICT.get(TAG_DESCONOCIDO, ["No sé cómo responder."]))
            bot_response_template = random.choice(bot_response_template_list)
            bot_suggestions = SUGGESTIONS_DICT.get(predicted_tag_final, [])
            
            final_bot_response = bot_response_template
            extracted_entities = {}
            calculation_results = {}

            if FULL_TRAINING_DATA:
                entidades_nlp_config = FULL_TRAINING_DATA.get('entidades_nlp', {})
                sistema_calculo_config = FULL_TRAINING_DATA.get('sistema_calculo_automatico', {})
                intent_config = next((i for i in FULL_TRAINING_DATA.get('intents', []) if i['tag'] == predicted_tag_final), None)

                if intent_config:
                    if 'entidades_extraer' in intent_config:
                        if predicted_tag_final == 'consulta_inteligente_limpieza':
                            extracted_entities = extraer_entidades_limpieza(cleaned_message, entidades_nlp_config)
                            print(f"DEBUG ChatAPIView: Entidades (Limpieza): {extracted_entities}")
                        # TODO: Añadir llamadas a otras funciones de extracción para otros intents

                    if 'funciones_callback' in intent_config:
                        if 'calcular_precio_limpieza' in intent_config['funciones_callback'] and predicted_tag_final == 'consulta_inteligente_limpieza':
                            calculation_results = calcular_precio_limpieza(extracted_entities, sistema_calculo_config)
                            print(f"DEBUG ChatAPIView: Cálculos (Limpieza): {calculation_results}")
                        # TODO: Añadir llamadas a otras funciones de cálculo

                    format_args = {}
                    if 'cantidad' in extracted_entities: format_args['cantidad'] = extracted_entities['cantidad']
                    if 'tipo_mueble_display' in extracted_entities: format_args['tipo_mueble'] = extracted_entities['tipo_mueble_display']
                    if 'calculo_detallado' in calculation_results: format_args['calculo_detallado'] = calculation_results['calculo_detallado']
                    
                    # TODO: Añadir más argumentos de formato según necesites
                    
                    try:
                        placeholders_in_template = re.findall(r'{(.*?)}', bot_response_template)
                        default_format_args = {ph: f"{{{ph}}}" for ph in placeholders_in_template}
                        for key, value in format_args.items():
                            if key in default_format_args: default_format_args[key] = value
                        final_bot_response = bot_response_template.format(**default_format_args)
                    except KeyError as e:
                        print(f"WARN ChatAPIView: KeyError '{e}' formateando respuesta.")
                        final_bot_response = bot_response_template
            else:
                print("WARN ChatAPIView: FULL_TRAINING_DATA no cargado. Respuesta genérica.")

            print(f"DEBUG ChatAPIView: Tag: '{predicted_tag_final}', Respuesta Final: '{final_bot_response}', Sugerencias: {bot_suggestions}")
            return JsonResponse({'response': final_bot_response, 'suggestions': bot_suggestions})

        except Exception as e:
            print(f"ERROR ChatAPIView: Excepción general: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': 'Error interno inesperado.', 'suggestions': []}, status=500)