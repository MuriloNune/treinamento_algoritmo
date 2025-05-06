📑 Sistema de Classificação e Comparação de Diários Oficiais
🔍 Objetivo:
Este projeto automatiza a classificação de páginas de PDFs e a extração de informações (como nomes de secretários e subprefeitos), comparando mudanças entre versões antigas e novas.

⚙️ Funcionalidades
✔ Classificação automática de páginas em:

frente (capa)

verso (contra-capa)

nenhum (conteúdo)

✔ Detecção de regiões de interesse (como campos de nomes) usando visão computacional.
✔ Extrai texto das áreas identificadas com OCR (pytesseract).
✔ Compara mudanças entre PDFs antigos e novos (ex: novos secretários, remoções).
✔ Download automático do PDF mais recente via Selenium.

📂 Estrutura dos Arquivos
📂 projeto-diario-oficial/  
├── 📄 comparar.py              # Pipeline completo (download, classificação, extração e comparação)  
├── 📄 algoritimo_achar_pag.py  # Modelo 1: Classificador de páginas (frente/verso/nenhum)  
├── 📄 algoritmo2.py            # Modelo 2: Detector de regiões para recorte  
├── 📂 dados/                   # Pastas de treinamento (opcional, se incluído)  
│   ├── 📂 Historico_imagem_treinamento/  # Imagens para treinar o classificador  
│   ├── 📂 cortado/             # Exemplos de imagens já recortadas  
│   └── 📂 n_cortado/           # Imagens completas sem recorte  
└── 📂 modelos/                 # Modelos pré-treinados (gerados pelos scripts)  
    ├── 📄 modelo_treinado.h5   # Modelo de classificação de páginas  
    └── 📄 modelo2_treinado.h5  # Modelo de detecção de regiões  
🚀 Como Usar
1. Pré-requisitos
Python 3.8+

Bibliotecas:

bash
pip install tensorflow pillow numpy pytesseract pdf2image selenium pyautogui opencv-python  
Tesseract OCR instalado (Download aqui).

2. Configuração
Baixe os modelos pré-treinados ou gere os seus:

Execute algoritimo_achar_pag.py para gerar modelo_treinado.h5 (classificação).

Execute algoritmo2.py para gerar modelo2_treinado.h5 (detecção de regiões).

Prepare os PDFs:

Coloque PDFs antigos em PDF_antigo/.

O script comparar.py baixará o novo PDF automaticamente.

3. Execução
bash
python comparar.py  
Saída esperada:

A(s) frente(s) é(são) a(s) página(s): 1, 2  
A(s) verso(s) é(são) a(s) página(s): 19, 20  
Mudanças detectadas:  
- Nome: João Silva | Antigo: Secretário | Novo: Subprefeito
  
🤖 Tecnologias Utilizadas
Machine Learning:

TensorFlow/Keras (redes neurais VGG16 + camadas personalizadas).

Data augmentation para melhor generalização.

Processamento de Imagens:

OpenCV/Pillow (redimensionamento, recorte).

pytesseract (OCR para extração de texto).

Automação:

Selenium (download automático de PDFs).

PyAutoGUI (interação com janelas de diálogo).

📌 Observações
Para treinar seus próprios modelos, organize os dados conforme a estrutura acima.

Ajuste hiperparâmetros (como learning_rate ou batch_size) se a precisão for baixa.

O modelo de classificação foi treinado com transfer learning (VGG16), enquanto o modelo de recorte usa uma abordagem binária (cortado vs. não cortado).
