import pickle
import pyLDAvis.gensim
import webbrowser
import os

lda = pickle.load(open('../output/lda_model.pickle', 'rb'))

vis = pyLDAvis.gensim.prepare(lda['model'], lda['corpus'], dictionary=lda['dictionary'])
pyLDAvis.save_html(vis, '../output/lda.html')

webbrowser.open(os.path.abspath('../output/lda.html'), new=2)