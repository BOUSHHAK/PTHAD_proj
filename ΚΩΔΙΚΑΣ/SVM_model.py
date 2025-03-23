# Εισαγωγη βιβλιοθηκων
import pandas as pd
import re
import time
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import sklearn.svm as sksvm
import joblib
# Για την προεπεξεργασια των reviews
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Για την μοντελοποιηση και την αξιολογιση του μοντελου
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Εφαρμογη TfidfVectorizer για αριθμητικη αναπαρασταση των reviews και η εκπαιδευση του γινεται πανω στο train set και επειτα εφαρμοζεται και στο test set.
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Εισαγωγη SVM μοντελου με χρηση linear kernel και στην συνεχεια η εκπαιδευση του και ο υπολογισμος του χρονου εκπαιδευσης
svm = sksvm.SVC(kernel='linear', C=1)
SVMtrainingTime = time.time()
svm_history = svm.fit(X_train_vec, Y_train)
SVMtrainingTime = time.time() - SVMtrainingTime

# Δοκιμη μοντελου 
SVMpred = svm.predict(X_test_vec)

# Αποθηκευση μοντελου και vectorizer
joblib.dump(svm, "svmModel.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Δημιουργια συναρτησης για υπολογισμο χρονου ταξινομησης
def SVMpredictionTime(review):
    vectorized = vectorizer.transform([review])
    predTime = time.time()
    svm.predict(vectorized)
    predTime = time.time() - predTime
    
    return predTime

predTime = 0
reviews = X_test.sample(20) # τυχαιες επιλογες
# Υπολογισμος μεσου χρονου ταξινομησης για 20 τυχαια review
for review in reviews:
    Time = SVMpredictionTime(review)
    predTime += Time

SVM_AVGPredTime = predTime / len(reviews)

# Αποδοση μοντελου
print("++++++++  SVM Model  ++++++++")
print("Precision: ", precision_score(Y_test, SVMpred))
print("Recall:    ", recall_score(Y_test, SVMpred))
print("F1:        ", f1_score(Y_test, SVMpred))
print("Accuracy:  ", accuracy_score(Y_test, SVMpred))
print("\nSVM Training time: ", SVMtrainingTime, "secs")
print("SVM Average prediction time of 20 random reviews:", SVM_AVGPredTime, "secs")