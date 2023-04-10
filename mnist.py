import numpy as np
import pandas as pd
from tensorflow import keras
train_data=pd.read_csv("C:\\Users\ADITHYA\Mnist\\train.csv")
train_labels=train_data.iloc[:,0].values.astype('int32')
train_images=train_data.iloc[:,1:].values.astype('float32')
train_images=train_images.reshape(train_images.shape[0],28,28)

import matplotlib.pyplot as plt
train_labels=keras.utils.to_categorical(train_labels)
from tensorflow.keras import layers,Sequential
model=Sequential([
    layers.Dense(units=256,activation='relu',input_shape=train_images[0].shape),
    layers.Dropout(0.1),
    layers.Dense(units=64,activation='relu'),
    layers.Dropout(0.1),

    layers.Dense(units=32,activation='relu'),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(units=10,activation='softmax')
])
model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
test_data=pd.read_csv("C:\\Users\ADITHYA\Mnist\\train.csv")
test_labels=test_data.iloc[:,0].values.astype('int32')
test_images=test_data.iloc[:,1:].values.astype('float32')
test_images=test_images.reshape(train_images.shape[0],28,28)
test_labels=keras.utils.to_categorical(test_labels)
model.fit(train_images,train_labels,epochs=10,batch_size=64,validation_split=0.2)
pred=model.predict(test_images, verbose=0)

pred=np.argmax(pred,axis=1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),
                          "Label": pred})
submissions.to_csv("submission.csv", index=False, header=True)