# Εισαγωγη βιβλιοθηκων
# Για χρηση αρχειων
import joblib
# Για δημιουργια web app
from flask import Flask, request, render_template
# Για χρηση του αποθηκευμενου μοντελου lstm
#from tensorflow.keras.models import load_model
# Για επεξεργασια του νεου review
#from tensorflow.keras.preprocessing.sequence import pad_sequences
# Για προεπεξεργασια review
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt') # για χρηση word_tokenize
nltk.download('stopwords') # για χρηση stopwords

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ========== Για SVM ===========
# Χρηση του μοντελου και του vectorizer
svm = joblib.load("svmModel.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ========== Για LSTM ==========
"""
# Χρηση του μοντελου και του tokenizer
lstm = load_model('lstmModel.keras')
tokenizer = joblib.load("tokenizer.pkl")
"""

# Συναρτηση προεπεξεργασιας review
def preprocess(text):
    text = re.sub(r'<.*?>', ' ', text) # Γινεται αφαιρεση των html tags και το περιεχομενο τους απο τα reviews
    tokens = word_tokenize(text.lower()) # Μετατροπη λεξεων σε πεζα και διαχωρισμος προτασεων σε μεμονωμενες λεξεις
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words] # Γινεται αφαιρεση των μη αλφαριθμητικων λεξεων και των stopwords
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens] # Γινεται εφαρμογη του stemmer για την επαναφορα των λεξεων στην κανονικη μορφη τους
    return ' '.join(stemmed_tokens) # Επιστρεφονται οι λεξεις με κενα αναμεσα τους

# Δημιουργια συναρτησης για την ταξινομηση του νεου review
def prediction(review):
    # ========== Για SVM ==========
    # Εφαρμογη της συνασρητσης preprocess
    preprocessed = preprocess(review)
    # Δημιουργια αριθμητικων παραστασεων
    vectorized = vectorizer.transform([preprocessed])
    # Ταξινομηση νεου review με το SVM μοντελο
    prediction = svm.predict(vectorized)
    if prediction[0] == 1 :
        sentiment = "positive"
    else:
        sentiment = "negative"
    return sentiment

    # ========== Για LSTM ==========
    """
    # Εφαρμογη της συνασρητσης preprocess
    preprocessed = preprocess(review)
    # Δημιουργια ακολουθιων και επεξεργασια στο μηκος τους
    padded_sequence = pad_sequences(tokenizer.texts_to_sequences([preprocessed]), padding='post', maxlen = 100)
    # Ταξινομηση νεου review με το LSTM μοντελο
    prediction = lstm.predict(padded_sequence)
    if prediction[0][0] > 0.5 :
        sentiment = "positive"
    else:
        sentiment = "negative"
    return sentiment
    """
    

# Δημιουργια της flask εφαρμογης
app = Flask(__name__)

# Αρχικη σελιδα του web app
@app.route('/')
def home():
    # Επιστρεφεται το template home.html ως αρχικη σελιδα
    return render_template('home.html')

# Διαδρομη για την ταξινομηση του νεου review
@app.route('/Sentiment', methods=['POST'])
def predict():
    # Ελεγχεται αν η μεθοδος ειναι POST
    if request.method == "POST":
        # Παιρνει το νεο review απο την φορμα του html
        review = request.form['review']
        # Γινεται χρηση της συναρτησης prediction για να γινει η ταξινομηση του νεου review
        sentiment = prediction(review)
        # Αναλογα την ταξινομηση οριζεται η εικονα με το αναλογο συναισθημα 
        if sentiment == "positive":
            image = "positive.png"
        else: 
            image = "negative.png"
    # Επιστρεφεται το template home.html μαζι με την αναλογη εικονα        
    return render_template('home.html', message=image)

# Γινεται η εκτελεση του web app
if __name__ == '__main__':
    # Ενεργοποιειται το debug και οριζεται η πυλη
    app.run(debug = True,port=8000)