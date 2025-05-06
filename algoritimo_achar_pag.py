import os  
import pathlib
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from pdf2image import convert_from_path

caminho_pdf = 'C:\\Users\\PICHAU\\Downloads\\comparar_pdf-20241021T031457Z-001\\comparar_pdf\\PDF_novo'
pdf_path = pathlib.Path(caminho_pdf) / 'Novo_Diario_Oficial.pdf'

caminho_treinamento = 'C:\\Users\\PICHAU\\Downloads\\comparar_pdf-20241021T031457Z-001\\comparar_pdf\\Historico_imagem_treinamento'
caminho_validacao = 'C:\\Users\\PICHAU\\Downloads\\comparar_pdf-20241021T031457Z-001\\comparar_pdf\\Historico_imagem_validação'

nova_largura = 675
nova_altura = 925

def extrair_imagens_pdf(pdf_path):
    imagens = convert_from_path(pdf_path, first_page=1, last_page=20)
    for i, img in enumerate(imagens):
        img = img.resize((nova_largura, nova_altura))
        img.save(f'pagina_{i + 1}.jpg', 'JPEG')

extrair_imagens_pdf(pdf_path)

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,         
    width_shift_range=0.2,     
    height_shift_range=0.2,    
    shear_range=0.2,           
    zoom_range=0.2,         
    horizontal_flip=True,     
    fill_mode='nearest'        
)

gerador_imagens_treinamento = datagen.flow_from_directory(
    caminho_treinamento,
    target_size=(nova_altura, nova_largura),
    class_mode='categorical',  
    classes=['frente', 'verso', 'nenhum'],
    batch_size=8
)

gerador_imagens_validacao = datagen.flow_from_directory(
    caminho_validacao,
    target_size=(nova_altura, nova_largura),
    class_mode='categorical', 
    classes=['frente', 'verso', 'nenhum'], 
    batch_size=8
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(nova_altura, nova_largura, 3))

for layer in base_model.layers:
    layer.trainable = True  

modelo = models.Sequential()
modelo.add(base_model)  
modelo.add(layers.Flatten())  
modelo.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))) 
modelo.add(layers.Dropout(0.5))  
modelo.add(layers.Dense(3, activation='softmax'))  

modelo.compile(optimizer=Adam(learning_rate=0.00001),
               loss='categorical_crossentropy',
               metrics=['accuracy'])


historico = modelo.fit(
    gerador_imagens_treinamento,
    epochs=30,
    validation_data=gerador_imagens_validacao
)

caminho_modelo = r'caminho onde quer salvar o modelo'
os.makedirs(caminho_modelo, exist_ok=True)  
modelo.save(os.path.join(caminho_modelo, 'modelo_treinado.h5')) 

def prever_imagem(img_path):
    img = image.load_img(img_path, target_size=(nova_altura, nova_largura))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = modelo.predict(img_array)
    return np.argmax(preds[0]) 

def comparar_imagens_pdf(num_paginas):
    frente_paginas = []
    verso_paginas = []
    nenhum_paginas = []

    for i in range(num_paginas):
        img_path = f'pagina_{i + 1}.jpg'
        classe = prever_imagem(img_path)

        if classe == 0:  
            frente_paginas.append(i + 1)
        elif classe == 1: 
            verso_paginas.append(i + 1)
        else:  
            nenhum_paginas.append(i + 1)

    if frente_paginas:
        print(f"A(s) frente(s) é(são) a(s) página(s): {', '.join(map(str, frente_paginas))}")
    else:
        print("Nenhuma página classificada como frente.")

    if verso_paginas:
        print(f"A(s) verso(s) é(são) a(s) página(s): {', '.join(map(str, verso_paginas))}")
    else:
        print("Nenhuma página classificada como verso.")

    if nenhum_paginas:
        print(f"As páginas descartadas (nenhum) são: {', '.join(map(str, nenhum_paginas))}")
    else:
        print("Nenhuma página descartada.")

comparar_imagens_pdf(20) 
