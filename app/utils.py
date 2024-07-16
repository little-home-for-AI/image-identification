from PIL import Image
import numpy as np
import tensorflow as tf

# 确保路径正确
model_path = 'models/cifar10_cnn.h5'
model = tf.keras.models.load_model(model_path)

# 在加载后编译模型
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 读取并预处理图片
def load_image(img_path):
    img = Image.open(img_path)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 预测函数
def predict_image(img_array):
    prediction = model.predict(img_array)
    return np.argmax(prediction[0])
