# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, SpatialDropout1D
from keras.layers.embeddings import Embedding
import keras.utils as kutil 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_X = pd.read_csv('../input/train.tsv', delimiter='\t')
test_X = pd.read_csv('../input/test.tsv', delimiter='\t')

phraseId = test_X.PhraseId

y_train = train_X['Sentiment'] #create dependent variable for applying classification
binary_train = kutil.to_categorical(y_train)

train_X.drop(['PhraseId', 'SentenceId', 'Sentiment'], axis=1, inplace=True)
test_X.drop(['PhraseId', 'SentenceId'], axis=1, inplace=True)

vocabulary_size = 10000
tokenizer_train = Tokenizer(num_words=vocabulary_size)
tokenizer_train.fit_on_texts(train_X['Phrase'])
sequences_train = tokenizer_train.texts_to_sequences(train_X['Phrase'])
X_train = pad_sequences(sequences_train, maxlen=150)

#convert test data
tokenizer_test = Tokenizer(num_words=vocabulary_size)
tokenizer_test.fit_on_texts(test_X['Phrase'])
sequences_test = tokenizer_test.texts_to_sequences(test_X['Phrase'])
X_test = pad_sequences(sequences_test, maxlen=150)

#build the neural network
embedding_dim = 128
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=X_train.shape[1])) 
model.add(SpatialDropout1D(0.7))
model.add(LSTM(64,dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())
model.fit(X_train, np.array(binary_train), validation_split=0.4, epochs=10, batch_size=1024)

y_pred = model.predict(X_test)
y_pred_final = y_pred.argmax(1)
# Any results you write to the current directory are saved as output.
my_submission = pd.DataFrame({'PhraseId': phraseId, 'Sentiment': y_pred_final})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)