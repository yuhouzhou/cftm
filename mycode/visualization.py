import pickle
import pyLDAvis.gensim

lda = pickle.load(open('../output/lda_model', 'rb'))

vis = pyLDAvis.gensim.prepare(lda['model'], lda['corpus'], dictionary=lda['dictionary'])
pyLDAvis.save_html(vis, '../output/lda.html')