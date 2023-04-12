import nltk
from nltk.corpus import gutenberg
from gensim.models import Word2Vec

# Download the Gutenberg Corpus
nltk.download('gutenberg')

# Load the sentences from the corpus
sentences = gutenberg.sents()

# Train a Word2Vec model with default hyperparameters
model = Word2Vec(sentences, min_count=5, vector_size=100, workers=4)

# Evaluate the learned word embeddings using analogy tasks
print(model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))

# Evaluate the learned word embeddings using similarity scores
print(model.wv.similarity('dog', 'cat'))
