# Εισαγωγη βιβλιοθηκων
import pandas as pd
import numpy as np
import re
import time
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import joblib
# Για την προεπεξεργασια των reviews
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Για την μοντελοποιηση και την αξιολογιση του μοντελου
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

nltk.download('punkt') # για χρηση word_tokenize
nltk.download('stopwords') # για χρηση stopwords

# Εισαγωγη αρχειου δεδομενων
data = pd.read_csv('imdb_master.csv')

#print(data.info())
#Παρουσιαση περιεχομενου της στηλης label. unsup=50000 neg=25000 pos=25000
#sns.countplot(x = data['label'])
#plt.show()

# Γινεται χρηση μονο των στηλων review, label απο το αρχειο δεδομενων
data = data[['review', 'label']] 

# Γινεται χρηση μονο των κατηγοριων pos, neg
data = data[data['label'] != 'unsup']

# Αντικατασταση των κατηγοριων pos, neg σε 1, 0
data['label'] = data['label'].map({"pos": 1, "neg": 0})

# Εισαγωγη stemmer για μετατροπη λεξεων στην κανονικη μορφη τους και stopwords για αφαιρεση μη αναγκαιων λεξεων
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'<.*?>', ' ', text) # Γινεται αφαιρεση των html tags και το περιεχομενο τους απο τα reviews
    tokens = word_tokenize(text.lower()) # Μετατροπη λεξεων σε πεζα και διαχωρισμος προτασεων σε μεμονωμενες λεξεις
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words] # Γινεται αφαιρεση των μη αλφαριθμητικων λεξεων και των stopwords
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens] # Γινεται εφαρμογη του stemmer για την επαναφορα των λεξεων στην κανονικη μορφη τους
    return ' '.join(stemmed_tokens) # Επιστρεφονται οι λεξεις με κενα αναμεσα τους

# Εφαρμογη της συναρτησης preprocess πανω σε καθε review και στην συνεχεια ανανεωνεται το περιεχομενο του data['review']
data['review'] = [preprocess(review) for review in data['review']]

# Τα reviews οριζονται ως X και τα labels ως Y
X = data['review']
Y = data['label']

# Τυχαιος διαχωρισμος του training και test set σε 80% και 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

# Δημιουργια tokenizer και εκπαιδευση του πανω στο train set
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Συνολικο πληθος μοναδικων λεξεων
vocab_length = len(word_index) + 1
#print(vocab_length)  #62873 unique words, test_size=0.2

# Αριθμος λεξεων σε καθε ακολουδια
max_words = 100

# Δημιουργια ακολουθιων αριθμων και επεξεργασια στο μηκος τους
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), padding='post', maxlen=max_words)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), padding='post', maxlen=max_words)

# split των reviews για χρηση word2vec
tokenized_corpus = [review.split() for review in data['review']]

# Εκπαιδευση μοντελου για δημιουργια word embeddings
model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,           
    window=5,                  
    min_count=1,               
    workers=4,                 
    sg=0                       
)
# Αποθηκευση του μοντελου word2vec
model.save("word2vec.model")
# Φορτωση word2vec
loaded_model = Word2Vec.load("word2vec.model")

# Διαστασεις των embeddings
embedding_dim = 100 

# Δημιουργια embedding matrix για οσες λεξεις υπαρχουν στο μοντελο word2vec 
embedding_matrix_vocab = np.zeros((vocab_length, embedding_dim))
for word, i in word_index.items():
    if word in loaded_model.wv:
        embedding_matrix_vocab[i] = loaded_model.wv[word]

# ========== LSTM MODEL ===========
# Εισοδος μηκους max_words
lstm = Sequential(name = "LSTM_Model")
lstm.add(Input(shape=(max_words,)))
lstm.add(Embedding(input_dim=vocab_length, output_dim=embedding_dim, weights = [embedding_matrix_vocab], trainable = False))
lstm.add(LSTM(128, return_sequences=False))
lstm.add(Dense(1, activation='sigmoid'))
lstm.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
lstm.summary()

# Εκπαιδευση μοντελου και υπολογισμους χρονου εκπαιδευσης
LSTMtrainingTime = time.time()
lstm_history = lstm.fit(X_train_seq, Y_train, batch_size = 128, epochs = 6, verbose = 1, validation_split = 0.2)
LSTMtrainingTime = time.time() - LSTMtrainingTime

# Αν οι προβλεψεις του μοντελου εχουν τιμες πανω απο 0,5 παιρνουν τιμη True, δηλαδη 1. Και False 0 για < 0.5
LSTMpred = (lstm.predict(X_test_seq) > 0.5).astype(int)

# Αποθηκευση μοντελου και tokenizer
lstm.save("lstmModel.keras")
joblib.dump(tokenizer, "tokenizer.pkl")

plt.plot(lstm_history.history['accuracy'])
plt.plot(lstm_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(lstm_history.history['loss'])
plt.plot(lstm_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

# Δημιουργια συναρτησης για υπολογισμο χρονου ταξινομησης
def LSTMpredictionTime(review):
    sequence = pad_sequences(tokenizer.texts_to_sequences([review]), padding='post', maxlen=max_words)
    LSTMpredTime = time.time()
    lstm.predict(sequence)
    LSTMpredTime = time.time() - LSTMpredTime

    return LSTMpredTime

predTime = 0
reviews = X_test.sample(20) # τυχαιες επιλογες
# Υπολογισμος μεσου χρονου ταξινομησης για 20 τυχαια review
for review in reviews:
    Time = LSTMpredictionTime(review)
    predTime += Time
LSTM_AVGPredTime = predTime / len(reviews)

# Αποδοση μοντελου
print("++++++++  LSTM Model  ++++++++")
print("Precision: ", precision_score(Y_test, LSTMpred))
print("Recall:    ", recall_score(Y_test, LSTMpred))
print("F1:        ", f1_score(Y_test, LSTMpred))
print("Accuracy:  ", accuracy_score(Y_test, LSTMpred))
print("\nLSTM Training time: ", LSTMtrainingTime, "secs")
print("LSTM Average prediction time of 20 random reviews:", LSTM_AVGPredTime, "secs")