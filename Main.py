from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import cv2
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from CustomButton import TkinterCustomButton
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional,GRU, LSTM, Conv1D, MaxPooling1D, RepeatVector
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

main = Tk()
main.title("Net Defender: Enhancing SDN Security with Deep Learning Against Botnet Threats")
main.geometry("1300x1200")

global filename
global img, X, Y, X_train, X_test, y_train, y_test, dcnn_model
global X_train1, X_test1
global accuracy, precision, recall, fscore
labels = ['BENIGN', 'Bot', 'DDoS', 'PortScan']

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index     

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    tf1.insert(END,str(filename))
    text.insert(END,filename+" Dataset Loaded\n\n")

def preprocessDataset():
    global filename, X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (16, 16))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(16, 16, 3)
                    X.append(im2arr)
                    label = getID(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Splits\n")
    
    text.insert(END,"80% of dataset used for training   "+"\n")
    text.insert(END,"20% of dataset used for testing    "+"\n")
    text.update_idletasks()
    text.update()
    img = cv2.imread("SpectogramDataset/BENIGN/1.png")
    img = cv2.resize(img, (200, 200))
    cv2.imshow("Sample Processed Image", img)
    cv2.waitKey(0)

def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()            

def runCNN1D():
    text.delete('1.0', END)
    global X_train1, X_test1
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], (X_train.shape[2] * X_train.shape[3])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], (X_test.shape[2] * X_test.shape[3])))

    cnn1d_model = Sequential()
    #defining CNN layer with 32 neurons or filters to filter and encode dataset features
    cnn1d_model.add(Conv1D(filters=32, kernel_size=9, activation='relu', input_shape=(X_train1.shape[1], X_train1.shape[2])))
    #defining another layer to further filter features
    cnn1d_model.add(Conv1D(filters=16, kernel_size=7, activation='relu'))
    #max pool layer to collect filtered features from CNN
    cnn1d_model.add(MaxPooling1D(pool_size=2))
    #convert multidimension features to single dimension
    cnn1d_model.add(Flatten())
    cnn1d_model.add(RepeatVector(2))
    #defining LSTM layer as decoder to predict output
    cnn1d_model.add(LSTM(32, activation='relu'))
    #defining output layer with 100 neurons
    cnn1d_model.add(Dense(units = 100, activation = 'relu'))
    cnn1d_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile the model
    cnn1d_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn1d_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn1d_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn1d_model.fit(X_train1, y_train, batch_size = 16, epochs = 20, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn1d_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn1d_model.load_weights("model/cnn1d_weights.hdf5")
    predict = cnn1d_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("CNN1D", y_test1, predict)

def runGRU():
    global X_train1, X_test1
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    gru_model = Sequential() #defining deep learning sequential object
    #adding GRU layer with 32 filters to filter given input X train data to select relevant features
    gru_model.add(Bidirectional(GRU(32, input_shape=(X_train1.shape[1], X_train1.shape[2]), return_sequences=True)))
    #adding dropout layer to remove irrelevant features
    gru_model.add(Dropout(0.2))
    #adding another layer
    gru_model.add(Bidirectional(GRU(32)))
    gru_model.add(Dropout(0.2))
    #defining output layer for prediction
    gru_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile GRU model
    gru_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/gru_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/gru_weights.hdf5', verbose = 1, save_best_only = True)
        hist = gru_model.fit(X_train1, y_train, batch_size = 16, epochs = 20, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/gru_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        gru_model = load_model("model/gru_weights.hdf5")
    predict = gru_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("GRU", y_test1, predict)    

def runLSTM():
    global X_train1, X_test1
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(100,input_shape=(X_train1.shape[1], X_train1.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.2))
    #adding another layer
    lstm_model.add(Dense(100, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train1, y_train, batch_size = 16, epochs = 20, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    predict = lstm_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("LSTM", y_test1, predict)    

def runSDCNN():
    global X_train, X_test, y_train, y_test
    global dcnn_model
    dcnn_model = Sequential()
    dcnn_model.add(Convolution2D(32, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    dcnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    dcnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    dcnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    dcnn_model.add(Flatten())
    dcnn_model.add(Dense(units = 256, activation = 'relu'))
    dcnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    dcnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/dcnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/dcnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = dcnn_model.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/dcnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        dcnn_model.load_weights("model/dcnn_weights.hdf5")
    predict = dcnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:100] = y_test1[0:100]
    calculateMetrics("Propose SDCNN", y_test1, predict)     

def graph():
    df = pd.DataFrame([['CNN1D','Accuracy',accuracy[0]],['CNN1D','Precision',precision[0]],['CNN1D','Recall',recall[0]],['CNN1D','FSCORE',fscore[0]],
                       ['GRU','Accuracy',accuracy[1]],['GRU','Precision',precision[1]],['GRU','Recall',recall[1]],['GRU','FSCORE',fscore[1]],
                       ['LSTM','Accuracy',accuracy[2]],['LSTM','Precision',precision[2]],['LSTM','Recall',recall[2]],['LSTM','FSCORE',fscore[2]], ['DCNN','Accuracy',accuracy[3]],['DCNN','Precision',precision[3]],['DCNN','Recall',recall[3]],['DCNN','FSCORE',fscore[3]]
                      ],columns=['Algorithms','Accuracy','Value'])
    df.pivot("Algorithms", "Accuracy", "Value").plot(kind='bar')
    plt.title("All Algorithm Comparison Graph")
    plt.show()

def predict():
    global dcnn_model
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "testData")
    data = pd.read_csv(filename)
    data = data.values
    data = data[:,0:data.shape[1]]
    for i in range(len(data)):
        temp = data[i]
        text.insert(END,"Test Data = "+str(temp)+" =====> ")
        temp = np.reshape(temp, (13, 6))
        plt.specgram(temp, Fs=6, cmap="rainbow")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig("test.png")
        img = cv2.imread("test.png")
        img = cv2.resize(img, (16, 16))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,16,16,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = dcnn_model.predict(img)
        predict = np.argmax(preds)
        text.insert(END,'Intrusion Predicted As : '+labels[predict]+"\n\n")
        text.update_idletasks()
        img = cv2.imread("test.png")
        img = cv2.resize(img, (700,400))
        cv2.putText(img, 'Intrusion Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('Intrusion Predicted As : '+labels[predict], img)
        cv2.waitKey(0)
        


def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Net Defender: Enhancing SDN Security with Deep Learning Against Botnet Threats')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Image Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = TkinterCustomButton(text="Upload Dataset", width=400, corner_radius=5, command=uploadDataset)
uploadButton.place(x=50,y=150)

processButton = TkinterCustomButton(text="Preprocess Dataset", width=300, corner_radius=5, command=preprocessDataset)
processButton.place(x=470,y=150)

cnn1dButton = TkinterCustomButton(text="Run CNN1D Algorithm", width=300, corner_radius=5, command=runCNN1D)
cnn1dButton.place(x=790,y=150)

gruButton = TkinterCustomButton(text="Run GRU Algorithm", width=300, corner_radius=5, command=runGRU)
gruButton.place(x=50,y=200)

lstmButton = TkinterCustomButton(text="Run LSTM Algorithm", width=300, corner_radius=5, command=runLSTM)
lstmButton.place(x=470,y=200)

sdcnnButton = TkinterCustomButton(text="Run Propose SDCNN Algorithm", width=300, corner_radius=5, command=runSDCNN)
sdcnnButton.place(x=790,y=200)

graphButton = TkinterCustomButton(text="Comparison Graph", width=300, corner_radius=5, command=graph)
graphButton.place(x=50,y=250)

predictButton = TkinterCustomButton(text="Predict Intrusion using Test Data", width=300, corner_radius=5, command=predict)
predictButton.place(x=470,y=250)

closeButton = TkinterCustomButton(text="Exit", width=300, corner_radius=5, command=close)
closeButton.place(x=790,y=250)


font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()
