from gensim.utils import SaveLoad
import pyLDAvis.gensim

lda = SaveLoad.load('./lda_model')

# Todo: import corpus and dictionary
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary=dictionary)