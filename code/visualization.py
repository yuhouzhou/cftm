import pickle
import pyLDAvis.gensim

lda = pickle.load(open('../model/lda_model', 'rb'))

# Todo: import corpus and dictionary
vis = pyLDAvis.gensim.prepare(lda['model'], lda['corpus'], dictionary=lda['dictionary'])