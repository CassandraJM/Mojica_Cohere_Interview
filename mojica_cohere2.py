import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

# Define the categories of emotions we want to detect
categories = ['neg', 'pos']

# Prepare the data for training and testing our model
data = []
for category in categories:
    for fileid in movie_reviews.fileids(category):
        words = movie_reviews.words(fileid)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
        bigrams = list(nltk.bigrams(words))
        data.append((words + bigrams, category))

random.shuffle(data)

train_data = data[:1500]
test_data = data[1500:]

# Define the features for the model
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % str(word)] = (word in document_words)
    return features

# Extract the features from the training data
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words() if w.isalpha() and w.lower() not in stopwords.words('english'))
word_features = list(all_words)[:2000]
bigram_features = list(nltk.FreqDist(nltk.bigrams(movie_reviews.words())))[2000:3000]
word_features += bigram_features

training_features = nltk.classify.apply_features(extract_features, train_data)

# Create a sentiment analyzer object
sentiment_analyzer = SentimentAnalyzer()

# Train the sentiment analyzer using the SVM classifier
print("training model")
trainer = SklearnClassifier(SVC(kernel='linear'))
classifier = trainer.train(training_features)

# Evaluate the performance of the model on the testing data
print("testing model")
testing_features = nltk.classify.apply_features(extract_features, test_data)
accuracy = nltk.classify.accuracy(classifier, testing_features)
print('Accuracy:', accuracy)

# Function to map sentiment output to emotion categories
def sentiment_to_emotion(sentiment):
    if sentiment == 'pos':
        return 'Positive'
    elif sentiment == 'neg':
        return 'Negative'
    else:
        return 'Unknown / Unable to detect emotion'

# Classify a new text and detect the emotions in it
text = 'I am so happy today!'
words = word_tokenize(text)
tagged_words = nltk.pos_tag(words)
features = extract_features(tagged_words)
sentiment_label = classifier.classify(features)
emotion_label = sentiment_to_emotion(sentiment_label)

print('Text:', text)
print('Emotion:', emotion_label)