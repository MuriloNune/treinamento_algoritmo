import os
import tempfile
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from pdf2image import convert_from_path
import pytesseract
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep

modelo1_path = r'C:caminho para o modelo'  # Modelo para identificar as páginas
modelo2_path = r'C:caminho para o modelo'  # Modelo para cortar a imagem

modelo1 = load_model(modelo1_path)
modelo2 = load_model(modelo2_path)

def prever_pagina(img_array):
    img_array = np.expand_dims(img_array, axis=0) 
    preds = modelo1.predict(img_array)
    return np.argmax(preds[0])  

def cortar_imagem(img_array):
    predicao = modelo2.predict(img_array[np.newaxis, ...])  

    x, y, largura, altura = np.squeeze(predicao).astype(int)  
    # Realizar o corte
    img_cortada = img_array[y:y + altura, x:x + largura] 
    return Image.fromarray((img_cortada * 255).astype(np.uint8))  

def extrair_texto_da_imagem(img):
    texto = pytesseract.image_to_string(img, lang='por')  
    return texto

def baixar_pdf(nome_arquivo='Novo_Diario_Oficial.pdf'):
    download_directory = r"caminho para o pdf novo"
    os.makedirs(download_directory, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        prefs = {
            "download.default_directory": download_directory,
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externamente": True
        }

        options = webdriver.ChromeOptions()
        options.add_experimental_option("prefs", prefs)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        driver.get('caminho do site')

        button = driver.find_element(By.XPATH, 'preencha o XPath')
        button.click()
        sleep(5)

        try:
            pyautogui.click(x=500, y=500, button='right')  
            sleep(1)
            pyautogui.press('down', presses=2)
            pyautogui.press('enter')
            sleep(2)

            caminho_completo = os.path.join(download_directory, nome_arquivo)
            pyautogui.write(caminho_completo)
            sleep(0.5)
            pyautogui.press('enter')
            sleep(5)

            while not any(fname.endswith('.pdf') for fname in os.listdir(download_directory)):
                sleep(1)

            return [os.path.join(download_directory, f) for f in os.listdir(download_directory) if f.endswith('.pdf')]
        except Exception as e:
            print(f"Erro ao encontrar ou clicar no botão de download: {e}")
        finally:
            driver.quit()

def processar_pdf(pdf_path, numero_paginas=20):
    imagens = convert_from_path(pdf_path)
    funcionarios = {}

    for i, img in enumerate(imagens[:numero_paginas]):
        img_resized = img.resize((675, 925))  
        img_resized_array = keras_image.img_to_array(img_resized) / 255.0  
        img_resized_array = np.array(img_resized_array)  

        if prever_pagina(img_resized_array) in [0, 1]:  
            img_cortada = cortar_imagem(img_resized_array)

            texto = extrair_texto_da_imagem(img_cortada)
            funcionarios.update(extrair_secretarios_e_subprefeitos(texto))

    return funcionarios

def extrair_secretarios_e_subprefeitos(texto):
    funcionarios = {}
    linhas = texto.splitlines()
    for linha in linhas:
        if "Nome:" in linha:  
            nome = linha.split("Nome:")[1].strip()
            funcionarios[nome] = nome  
    return funcionarios

def comparar_funcionarios(antigos, novo):
    mudancas = {}
    for key in novo.keys():
        if key in antigos:
            if antigos[key] != novo[key]:
                mudancas[key] = {
                    "antigo": antigos[key],
                    "novo": novo[key]
                }
        else:
            mudancas[key] = {
                "antigo": None,
                "novo": novo[key]
            }
    return mudancas

# Caminho para os PDFs antigos
caminho_dos_pdfs_antigos = r"caminho do pdf antigo para o padrão de comparação"


funcionarios_antigos = {}
for pdf in os.listdir(caminho_dos_pdfs_antigos):
    pdf_path = os.path.join(caminho_dos_pdfs_antigos, pdf)
    funcionarios_antigos.update(processar_pdf(pdf_path, numero_paginas=20))

caminhos_novos_pdfs = baixar_pdf(nome_arquivo='Novo_Diario_Oficial.pdf')

if caminhos_novos_pdfs:
    novo_pdf_path = caminhos_novos_pdfs[0]

    funcionarios_novos = processar_pdf(novo_pdf_path, numero_paginas=20)

    mudancas = comparar_funcionarios(funcionarios_antigos, funcionarios_novos)

    for key, value in mudancas.items():
        print(f"{key}: Antigo: {value['antigo']}, Novo: {value['novo']}")

    os.remove(novo_pdf_path)
else:
    print("Nenhum PDF novo encontrado após o download.")
