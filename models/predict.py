import numpy as np
import tensorflow as tf
from PIL import Image

# 加载模型
model = tf.keras.models.load_model('models/cifar10_cnn.h5')

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

# 示例预测
if __name__ == '__main__':
    img_path = 'path_to_image'
    img_array = load_image(img_path)
    prediction = predict_image(img_array)
    print(f"The image is predicted to be a {class_names[prediction]}")
