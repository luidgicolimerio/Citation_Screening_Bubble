from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, SystemMessage
import PyPDF2

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('GROQ_API_KEY')

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.1-70b-versatile")

def citation_screening(text):
    system_prompt="""
        Você está conduzindo uma revisão sistemática e meta-análise,
        com foco em uma área específica do uso de ferramentas digitais.
        Você irá receber a primeira página de um artigo, você deve analisar
        o título e o abstract desse artigo.
        Sua tarefa é avaliar os estudos de pesquisa e listar as ferramentas digitais usadas em cada estudo. 
        Caracterizam-se como ferramentas digitais: ChatGPT, Google Gemini, Microsoft Copilot, WhatsApp, Python, PowerBI, Excel,
        Inteligência Artificial (IA), Machine Learning, Telemóvel, Internet, Sistema de Informação, Wi-Fi, Computador, Smartwatch, Internet das Coisas, Sensores Vestíveis,
        aplicativos ou aplicativos móveis.
        Por favor, responda apenas com as ferramentas utilizadas, uma após a outra, liste não mais do que as 5 principais aplicações utilizadas nos estudos,
        Se o estudo não usar nenhuma ferramenta digital, basta dizer que você não conseguiu identificar nenhuma.
        Sua resposta deve conter apenas as ferramentas seguindo o exemplo abaixo:
        O estudo utilizou:
        - Chat GPT
        - Python
        - Machine Learning
    """
    user_prompt = f"""
            {text}
        """
    menssages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    answer = llm.invoke(menssages)

    return answer.content

def paper_selection(text, exclude_criteria):
    answer = llm.invoke(f"""You are conducting a systematic review and meta-analysis in the medical area. Your task is to evaluate research studies and determine whether they
                            should be included in your review. To do this, each study must meet ALL the inclusion criteria and NONE of the exclusion criteria:
                # Inclusion criteria

                    - Studies that implemented eHealth interventions in primary care settings were conducted in LMICs.

                    - Studies focusing on eHealth interventions implemented specifically in primary care settings.

                    - Studies published in peer-reviewed journals.

                    - Studies reporting on the characteristics, outcomes, implementation processes, or evaluations of eHealth interventions.

                    - Studies involving healthcare providers, patients, or stakeholders directly involved in the delivery or utilisation of eHealth interventions in primary care settings.

                # Exclusion criteria
                    
                    {exclude_criteria}

                      
                 After reading the title and abstract of a study, you will decide whether to include or exclude it based on these criteria. ALWAYS initiate your answer with 'include' or 'exclude'. After that, justify your decision with a brief explanation.
                Paper:
                      {text}""")
    return answer.content

def extract_text_pypdf2(file):
    pdf_reader = PyPDF2.PdfReader(file)
   
    first_page_text = pdf_reader.pages[0].extract_text()
    return first_page_text

app = Flask(__name__)

@app.route('/citation_screening_upload', methods=['POST'])
def citation_screening_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    text = extract_text_pypdf2(file)
    anwser = citation_screening(text)

    return jsonify({'text': anwser})

@app.route('/paper_selection_upload', methods=['POST'])
def paper_selection_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    text = extract_text_pypdf2(file)

    exclude_criteria = request.form.get('exclude_criteria', '')
    anwser = paper_selection(text, exclude_criteria)

    return jsonify({'text': anwser[0:7]})

if __name__ == '__main__':
    app.run(debug=True)