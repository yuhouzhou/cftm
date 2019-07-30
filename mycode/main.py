import cftm_parser
import preprocessing as pp
from spacy.lang.de.stop_words import STOP_WORDS
from gensim.models.ldamodel import LdaModel
import pickle
import pyLDAvis.gensim
import webbrowser
import os

# Parsing
path1 = "../data.nosync/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
path2 = "../data.nosync/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
df_pd = cftm_parser.parquet_transform(path1, path2, n=20000)

# Preprocessing
stopwords = list(STOP_WORDS)
dictionary, corpus = pp.preprocessor(df_pd, stopwords=stopwords, language='de', text='TEXT', metadata='DATE',
                                     min_len=300)

# Data Modelling
lda = LdaModel(corpus, num_topics=10)
# pickle the model
lda_pickle = {"model": lda, "dictionary": dictionary, "corpus": corpus}
pickle.dump(lda_pickle, open('../output/lda_model.pickle', 'wb'))

# Data Visualization
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary=dictionary)
pyLDAvis.save_html(vis, '../output/lda.html')
webbrowser.open(os.path.abspath('../output/lda.html'), new=2)

# Data Evaluation
