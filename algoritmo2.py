import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Caminhos para os dados de treinamento
caminho_exemplos_corretos = r'caminho para os dados cortados'
caminho_imagens_completas = r'caminho para os dados não cortados'

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size 

print(f"Arquivos encontrados em '{caminho_exemplos_corretos}': {os.listdir(caminho_exemplos_corretos)}")
print(f"Arquivos encontrados em '{caminho_imagens_completas}': {os.listdir(caminho_imagens_completas)}")

image_files = [os.path.join(caminho_exemplos_corretos, fname) for fname in os.listdir(caminho_exemplos_corretos) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
if image_files:
    largura, altura = get_image_dimensions(image_files[0])
else:
    raise ValueError("Nenhuma imagem encontrada no diretório de exemplos corretos.")

print(f"Dimensões da imagem: largura={largura}, altura={altura}")

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

def load_images(path):
    images = []
    for fname in os.listdir(path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, fname)
            print(f"Carregando imagem: {img_path}") 
            img = load_img(img_path, target_size=(altura, largura))
            img_array = img_to_array(img)
            images.append(img_array)
    return np.array(images)

imagens_cortadas = load_images(caminho_exemplos_corretos)
print(f"Número de imagens cortadas carregadas: {len(imagens_cortadas)}")

imagens_n_cortadas = load_images(caminho_imagens_completas)
print(f"Número de imagens não cortadas carregadas: {len(imagens_n_cortadas)}")

rótulos_cortados = np.ones(len(imagens_cortadas))
rótulos_n_cortadas = np.zeros(len(imagens_n_cortadas))

imagens = np.concatenate((imagens_cortadas, imagens_n_cortadas), axis=0)
rótulos = np.concatenate((rótulos_cortados, rótulos_n_cortadas), axis=0)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(imagens, rótulos, test_size=0.2, random_state=42)

train_datagen = datagen.flow(X_train, y_train, batch_size=32)
val_datagen = datagen.flow(X_val, y_val, batch_size=32)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(altura, largura, 3))

for layer in base_model.layers:
    layer.trainable = True

modelo = models.Sequential()
modelo.add(base_model) 
modelo.add(layers.Flatten())  
modelo.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))  
modelo.add(layers.Dropout(0.5))  
modelo.add(layers.Dense(1, activation='sigmoid'))  

modelo.compile(optimizer=Adam(learning_rate=0.00001),
               loss='binary_crossentropy',
               metrics=['accuracy'])

historico = modelo.fit(
    train_datagen,
    epochs=30,
    validation_data=val_datagen
)

modelo.save(r'caminho onde quer salvar o modelo')

print("Treinamento concluído e modelo salvo.")
