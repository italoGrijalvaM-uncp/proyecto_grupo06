# 📩 Sistema Inteligente de Detección de Spam (Spambase)

Este repositorio contiene una solución completa de Machine Learning para la clasificación de correos electrónicos en **Spam** o **No Spam**, que incluye entrenamiento de red neuronal MLP, despliegue de una API RESTful en FastAPI, pruebas automáticas con pytest y un chatbot conversacional interactivo en Streamlit con soporte multilingüe y garantías de privacidad.

## 👥 Integrantes - Grupo 06
* Proyecto de Tecnologías Emergentes (ISO46B)

---Grijalva Martinez Italo Osmar

## 🛠️ Requisitos Previos e Instalación

### 1. Clonar el repositorio e ingresar a la carpeta:
```bash
git clone https://github.com/italoGrijalvaM-uncp/proyecto_grupo06
cd proyecto_grupo06
```
### 2. Crear y activar el entorno virtual (venv) para python >= 3.10
```Powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
en MAC 
```Powershell
python3 -m venv .venv
source .venv/bin/activate
```
### 3. instalar las dependencias del proyecto 
```powershell
pip install -r requirements.txt
pip install -e .
```
## Ejecución del Sistema 🚀
### Paso 1. Levantar el Backend (API de FastAPI)
```Powershell
uvicorn src.api:app --reload
```
Documentación Swagger interactiva: http://127.0.0.1:8000/docs

Endpoint de salud: http://127.0.0.1:8000/salud

### paso 2. Levantar el chatbot conversacional(Streamlit)
en otra terminal sin tocar la anterior
```Powershell
streamlit run src/chatbot.py
```
La interfaz web se abrirá automáticamente en http://localhost:8501
## Ejecución automática de calidad 🧪
Para verificar la integridad de los datos, los endpoints de la API y el chatbot:
```Powershell
pytest -v 
```
## Cuidado ético y privacidad
Este sistema cumple con estrictos principios de seguridad: el contenido de los correos procesados no se almacena en ninguna base de datos ni archivo persistente. El texto ingresado en el chatbot se traduce y analiza únicamente en memoria volátil (RAM) para la generación instantánea de métricas numéricas.
