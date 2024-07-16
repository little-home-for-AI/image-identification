import pickle
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# 解包数据集文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 使用绝对路径
data_path = os.path.abspath('data/cifar-10-batches-py/')


def load_data_file(file_path):
    batch = unpickle(file_path)
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    labels = np.array(labels)
    return data, labels


# 读取每个批次的数据
data_batch_1, labels_batch_1 = load_data_file(os.path.abspath('D:\PycharmProjects\image-identification\data\cifar-10'
                                                              '-batches-py\data_batch_1'))
data_batch_2, labels_batch_2 = load_data_file(
    os.path.abspath('D:\PycharmProjects\image-identification\data\cifar-10-batches-py\data_batch_2'))
data_batch_3, labels_batch_3 = load_data_file(
    os.path.abspath('D:\PycharmProjects\image-identification\data\cifar-10-batches-py\data_batch_3'))
data_batch_4, labels_batch_4 = load_data_file(
    os.path.abspath('D:\PycharmProjects\image-identification\data\cifar-10-batches-py\data_batch_4'))
data_batch_5, labels_batch_5 = load_data_file(
    os.path.abspath('D:\PycharmProjects\image-identification\data\cifar-10-batches-py\data_batch_5'))

# 合并所有训练数据
x_train = np.concatenate([data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5])
y_train = np.concatenate([labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5])

# 读取测试数据
test_batch = unpickle(os.path.abspath('D://PycharmProjects//image-identification//data//cifar-10-batches-py//test_batch'))
x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
y_test = np.array(test_batch[b'labels'])

# 构建CNN模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/cifar10_cnn.h5')

# 可视化训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()