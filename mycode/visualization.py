import pickle
import pyLDAvis.gensim

lda = pickle.load(open('../model/lda_model', 'rb'))

vis = pyLDAvis.gensim.prepare(lda['model'], lda['corpus'], dictionary=lda['dictionary'])