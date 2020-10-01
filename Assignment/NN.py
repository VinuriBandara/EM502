import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import datetime

from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


CSV_COLUMN_NAMES = ['Age', 'Gender', 'TB', 'DB', 'ALK', 'SGPT', 'SGOT', 'TP', 'ALB', 'AG_Ratio', 'Class']



train_data = pd.read_csv("training_data.csv", names=CSV_COLUMN_NAMES, header=0) 
test_data = pd.read_csv("testing_data.csv", names=CSV_COLUMN_NAMES, header=0) 



train_data=train_data.replace('?', np.NaN)
test_data=test_data.replace('?', np.NaN)



convert_dict = {
    'TB' : float,
    'DB' : float,
    'ALK' : float,
    'SGPT' : float,
    'SGOT' : float,
    'TP' : float,
    'ALB' : float,
    'AG_Ratio' : float
}

train_data = train_data.astype(convert_dict) 
test_data = test_data.astype(convert_dict)


MEAN = train_data.mean()



train_data_new = train_data.fillna(value=MEAN)



train_data_new["Gender"]=train_data_new["Gender"].replace('Female',1)
train_data_new["Gender"]=train_data_new["Gender"].replace('Male',0)



train_data_new["Class"]=train_data_new["Class"].replace({'Yes':1, 'No':0})


class0 = train_data_new["Class"] == 0
train_data_copy = train_data_new[class0]


train_data_new = train_data_new.append(train_data_copy, ignore_index=True)



train_data_new = train_data_new.sample(frac=1).reset_index(drop=True)


train_class = train_data_new.pop("Class")


MEAN_T = test_data.mean()


test_data_new = test_data.fillna(value=MEAN)


test_data_new["Gender"]=test_data_new["Gender"].replace('Female',1)
test_data_new["Gender"]=test_data_new["Gender"].replace('Male',0)


test_class = test_data_new.pop("Class")


test_class=test_class.replace({'Yes':1, 'No':0})



scaler = StandardScaler()
scaler.fit(train_data_new.values)

params = scaler.mean_


train_data_scaled = scaler.transform(train_data_new.values)
train_data_scaled = pd.DataFrame(train_data_scaled, index=train_data_new.index, columns=train_data_new.columns)


train_data_scaled.describe()



test_data_scaled = scaler.transform(test_data_new.values)
test_data_scaled = pd.DataFrame(test_data_scaled, index=test_data_new.index, columns=test_data_new.columns)


X_train = train_data_scaled.to_numpy()
y_train = train_class.to_numpy()


X_test = test_data_scaled.to_numpy()
y_test = test_class.to_numpy()


# neural network 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Activation, BatchNormalization

do = 0.5
model = Sequential()


model.add(Dense(50, input_dim=10))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(do))


model.add(Dense(50))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(do))

model.add(Dense(50))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(do))

model.add(Dense(20))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(do))


model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


from tensorflow.keras.callbacks import ModelCheckpoint


path = 'temp.h5'


keras_callbacks = [
       ModelCheckpoint(
            filepath= path,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1)
]


history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=10, verbose=1,
                    callbacks=keras_callbacks)



model.load_weights(path)


y_pred_ = model.predict(X_test)


ind0 = np.where(y_test==0)
ind1 = np.where(y_test==1)

y_pred_0 = y_pred_[ind0]
y_pred_1 = y_pred_[ind1]
y_test_0 = y_test[ind0] 
y_test_1 = y_test[ind1]

pyplot.figure()
pyplot.scatter(y_pred_0 ,y_test_0)
pyplot.scatter(y_pred_1, y_test_1)

pyplot.figure()
pyplot.hist(y_pred_0, 10, facecolor='blue', alpha=0.5)
pyplot.hist(y_pred_1, 10, facecolor='orange', alpha=0.5)
pyplot.show()


y_pred = np.round(y_pred_)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



len(np.where(y_pred == 1)[0]) #221


len(np.where(y_pred == 0)[0]) #90


pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'val'], loc='upper left')
pyplot.show()



confusion = metrics.confusion_matrix(y_test, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


accuracy = (TP + TN) / float(TP + TN + FP + FN)
classification_error = (FP + FN) / float(TP + TN + FP + FN)
sensitivity = TP / float(FN + TP)
specificity = TN / (TN + FP)
precision = TP / float(TP + FP)
false_positive_rate = FP / float(TN + FP)


print ('Accuracy : ',accuracy)
print ('Error rate : ',classification_error)
print ('Sensitvity : ',sensitivity)
print ('Specificity : ',specificity)
print ('Precision : ',precision)
print ('FPR : ',false_positive_rate)

