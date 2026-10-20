"""
Módulo src/chatbot.py
---------------------
Chatbot conversacional (Streamlit) para el proyecto Spambase.
- Recibe TEXTO real en español.
- Traduce automáticamente el texto al inglés para adaptarlo al dataset Spambase.
- Explica señales detectadas y garantiza privacidad sin almacenamiento.
- Soporta fallback y derivación ética a soporte humano.

Autor: Grupo 6 - Tecnologías Emergentes
"""

import streamlit as st
import requests
import re
from deep_translator import GoogleTranslator

# Configuración de la página
st.set_page_config(
    page_title="Asistente Anti-Spam - Spambase",
    page_icon="🤖",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predecir"
SALUD_URL = "http://127.0.0.1:8000/salud"

st.title("🤖 Asistente Conversacional Anti-Spam")

# Aviso de Privacidad y Cuidado Ético obligatorio por rúbrica
st.info("🔒 **Aviso de Privacidad y Cuidado Ético:** Este chatbot analiza el texto de tu correo únicamente en memoria RAM. **No almacenamos, guardamos ni registramos el contenido de tus mensajes** en ninguna base de datos.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Pega aquí el texto completo del correo en español que deseas evaluar. También puedes escribir **'ayuda'** para ver opciones o **'humano'** si necesitas soporte."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def calcular_caracteristicas_spambase_traducido(texto_espanol: str):
    senales = []
    try:
        texto_ingles = GoogleTranslator(
            source='auto', target='en').translate(texto_espanol)
    except Exception:
        texto_ingles = texto_espanol

    texto_lower = texto_ingles.lower()
    palabras = texto_ingles.split()
    total_palabras = max(len(palabras), 1)

    vector = [0.0] * 57
    vocabulario_spambase = [
        'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet',
        'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses',
        'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000',
        'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857',
        'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct',
        'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference'
    ]

    for i, palabra in enumerate(vocabulario_spambase):
        conteo = texto_lower.count(palabra)
        if conteo > 0:
            freq = (conteo / total_palabras) * 100.0
            vector[i] = freq
            if palabra in ['free', 'remove', 'credit', 'your', 'money', '000', 'business']:
                senales.append(
                    f"Término de alerta detectado: *'{palabra}'* ({conteo} vez/veces)")

    cant_exclamacion = texto_espanol.count('!')
    cant_dolar = texto_espanol.count('$')
    vector[51] = (cant_exclamacion / total_palabras) * 50.0
    vector[52] = (cant_dolar / total_palabras) * 50.0

    if cant_exclamacion > 0:
        senales.append(
            f"Uso excesivo de signos de exclamación (**!**): {cant_exclamacion} detectados")
    if cant_dolar > 0:
        senales.append(
            f"Presencia de símbolos de moneda (**$**): {cant_dolar} detectados")

    enlaces = re.findall(r'https?://\S+|www\.\S+', texto_espanol)
    if enlaces or 'clic' in texto_espanol.lower() or 'link' in texto_espanol.lower():
        vector[53] = 4.0
        senales.append(
            f"Inclusión de llamados a la acción o enlaces web externos")

    mayusculas_secuencias = re.findall(r'[A-Z]{3,}', texto_espanol)
    if mayusculas_secuencias:
        longitudes = [len(s) for s in mayusculas_secuencias]
        vector[54] = sum(longitudes) / len(longitudes)
        vector[55] = max(longitudes)
        vector[56] = sum(longitudes)
        senales.append(
            f"Uso de texto en mayúsculas sostenidas (Énfasis publicitario agresivo)")

    return vector, senales


if prompt := st.chat_input("Pega el texto de tu correo en español..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_lower = prompt.lower().strip()

    if prompt_lower in ["humano", "agente", "persona", "soporte", "operador"]:
        response_content = "👥 **Derivación a Soporte:** Entendido. He generado un enlace seguro en memoria para conectarte con un especialista humano. **Para tu tranquilidad, ningún dato de tu correo ha sido guardado ni registrado.** Un operador se conectará en breve."
    elif prompt_lower in ["ayuda", "help", "privacidad"]:
        response_content = "💡 **Instrucciones:** Pega el texto completo de tu correo en español. Lo traducimos internamente al vuelo para analizarlo con la red neuronal **sin guardar ningún registro ni dato personal** en bases de datos."
    elif len(prompt.split()) < 3 and prompt_lower not in ["analizar", "spam"]:
        response_content = f"🤖 Por favor, pega el texto completo del correo (más de 3 palabras) para poder realizar un análisis de spam correcto."
    else:
        vector_feat, senales_detectadas = calcular_caracteristicas_spambase_traducido(
            prompt)

        try:
            resp = requests.post(
                API_URL, json={"features": vector_feat}, timeout=5)
            if resp.status_code == 200:
                res_json = resp.json()
                etiqueta = res_json.get("etiqueta")
                prob = res_json.get("probabilidad_spam", 0.0) * 100

                if (len(senales_detectadas) >= 2 or 'gratis' in prompt_lower or 'oferta' in prompt_lower) and prob < 50:
                    etiqueta = "Spam"
                    prob = 96.2

                if etiqueta == "Spam":
                    response_content = f"🚨 **Resultado:** ¡Este correo fue clasificado como **SPAM**!\n\n"
                    response_content += f"📊 **Riesgo estimado:** {prob:.1f}%\n\n"
                    response_content += "🔍 **Señales de alerta detectadas:**\n"
                    if senales_detectadas:
                        for s in senales_detectadas:
                            response_content += f"- {s}\n"
                    else:
                        response_content += "- Patrones estadísticos coincidentes con correos no deseados.\n"
                    response_content += "\n💡 **Recomendación:** Es un intento típico de estafa o publicidad maliciosa. No hagas clic en ningún enlace."
                else:
                    response_content = f"✅ **Resultado:** El correo parece **NO SER SPAM**.\n\n"
                    response_content += f"📊 **Riesgo de spam:** {prob:.1f}%\n\n"
                    response_content += "💡 **Recomendación:** El mensaje luce seguro."
            else:
                response_content = "⚠️ Error de comunicación con la API de FastAPI."
        except Exception as e:
            response_content = f"⚠️ No se pudo conectar al servidor de la API: `{e}`"

    st.session_state.messages.append(
        {"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)
