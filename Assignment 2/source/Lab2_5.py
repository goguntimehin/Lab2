import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\Josh\Downloads\NBA_train.csv")
print(data.head())
data = data[['Team','W']]

data['Team'] = data['Team']
data['Team'] = data['Team']

max_features = 1000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['Team'])
X = tokenizer.texts_to_sequences(data['Team'].values)

X = pad_sequences(X)
epochs=1
embed_dim = 120
lstm_out = 100

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
      # return model
print(model.summary())



labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['Team'])
y = integer_encoded
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

batch_size = 32

hist=model.fit(X_train, Y_train, epochs = 1, batch_size=batch_size, verbose = 2)
scores = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
print("Lstm Accuracy: %.2f%%" % (scores[1]*100))
print(hist)
print(scores)
print(model.metrics_names)

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))



