import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 解包数据集文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 数据路径
data_path = 'D:/PycharmProjects/image-identification/cifar-10-batches-py/'

# 读取数据批次
def load_data_batch(batch_id):
    file_path = data_path + 'data_batch_' + str(batch_id)
    batch = unpickle(file_path)
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    labels = np.array(labels)
    return data, labels

# 读取所有训练数据
x_train, y_train = [], []
for i in range(1, 6):
    data, labels = load_data_batch(i)
    x_train.append(data)
    y_train.append(labels)
x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

# 读取测试数据
test_batch = unpickle(data_path + 'test_batch')
x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
y_test = np.array(test_batch[b'labels'])

# 构建CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test))

# 保存模型
model.save('cifar10_cnn.h5')

# 可视化训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from tensorflow.keras.preprocessing import image

# 加载模型
model = tf.keras.models.load_model('cifar10_cnn.h5')

# 读取并预处理图片
def load_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 预测函数
def predict_image(img_path):
    img_array = load_image(img_path)
    prediction = model.predict(img_array)
    return np.argmax(prediction[0])

# 示例预测
img_path = 'path_to_image'
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
prediction = predict_image(img_path)
print(f"The image is predicted to be a {class_names[prediction]}")
