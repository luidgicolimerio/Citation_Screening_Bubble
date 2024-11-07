from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, SystemMessage
import PyPDF2

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['GROQ_API_KEY']

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.1-70b-versatile")

def revisor_agent(text):
    system_prompt="""
        Você está conduzindo uma revisão sistemática e meta-análise,
        com foco em uma área específica do uso de ferramentas digitais.
        Você irá receber um texto de um artigo.
        Sua tarefa é avaliar os estudos de pesquisa e listar as ferramentas digitais usadas em cada estudo. 
        Caracterizam-se como ferramentas digitais: ChatGPT, Google Gemini, Microsoft Copilot, WhatsApp, Python, PowerBI, Excel,
        Inteligência Artificial (IA), Machine Learning, Telemóvel, Internet, Sistema de Informação, Wi-Fi, Computador, Smartwatch, Internet das Coisas, Sensores Vestíveis,
        Telemedicina, Teleconsulta, Telecomunicações, Telessaúde, aplicativos ou aplicativos móveis.
        Por favor, responda apenas com as ferramentas utilizadas, uma após a outra, liist não mais do que as 5 principais aplicações utilizadas nos estudos,
        Se o estudo não usar nenhuma ferramenta digital, basta dizer que você não conseguiu identificar nenhuma.
    """
    user_prompt = f"""
            {text}
        """
    menssages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    answer = llm.invoke(menssages)

    return answer.content

def extract_text_pypdf2(file):
    pdf_reader = PyPDF2.PdfReader(file)
   
    first_page_text = pdf_reader.pages[0].extract_text()
    return first_page_text

app = Flask(__name__)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    text = extract_text_pypdf2(file)
    anwser = revisor_agent(text)

    return jsonify({'text': anwser})

if __name__ == '__main__':
    app.run(debug=True)