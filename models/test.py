import tensorflow as tf

# 创建Sequential模型
model = tf.keras.Sequential()

# 添加层
model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=784))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型（假设x_train和y_train已经准备好）
model.fit(x_train, y_train, epochs=10, batch_size=32)

# 评估模型（假设x_test和y_test已经准备好）
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
