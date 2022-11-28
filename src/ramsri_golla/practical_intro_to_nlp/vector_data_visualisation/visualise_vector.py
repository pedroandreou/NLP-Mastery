from gensim.models import Word2Vec
import pandas as pd
import nltk

nltk.download("punkt")
import numpy as np
import matplotlib.pyplot as plt


text = """Word embedding is the collective name for a set of language modeling and feature learning techniques
in natural language processing where words or phrases from the vocabulary are mapped to vectors of real numbers.
Conceptually it involves a mathematical embedding from a space with many dimensions per word to a continuous
vector space with a much lower dimension.The use of multi-sense embeddings is known to improve performance in several NLP tasks,
such as part-of-speech tagging, semantic relation identification, and semantic relatedness. However,
tasks involving named entity recognition and sentiment analysis seem not to benefit from a multiple vector representation."""


# text = re.sub(r"[^.A-Za-z]", ' ', text)
sentence = text.split(".")
tokens = [nltk.word_tokenize(words) for words in sentence]


# size: the number of dimensions for the representation of words
# window: the max distance between a central word and words around the central word
# sg: training algorithm, 1=Skip-gram
# min_count: min occurence of word, otherwise it will be ignored
model = Word2Vec(tokens, size=50, sg=1, min_count=1)


# Store all the word vectors in a dataframe
X = model[model.wv.vocab]
df = pd.DataFrame(X)


### PCA ###
# Computing the correlation matrix
X_corr = df.corr()

# Computing eigen values and eigen vectors
values, vectors = np.linalg.eig(X_corr)

# Sorting the eigen vectors coresponding to eigen values in descending order
args = (-values).argsort()
values = vectors[args]
vectors = vectors[:, args]

# Taking first 2 components which explain maximum variance for projecting
new_vectors = vectors[:, :2]

# Projecting it onto new dimesion with 2 axis
neww_X = np.dot(X, new_vectors)


plt.figure(figsize=(13, 7))
plt.scatter(neww_X[:, 0], neww_X[:, 1], linewidths=10, color="blue")
plt.xlabel("PC1", size=15)
plt.ylabel("PC2", size=15)
plt.title("Word Embedding Space", size=20)

vocab = list(model.wv.vocab)

for i, word in enumerate(vocab):
    plt.annotate(word, xy=(neww_X[i, 0], neww_X[i, 1]))

plt.show()
