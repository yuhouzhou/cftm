import preprocessing as pp
from spacy.lang.de.stop_words import STOP_WORDS
from gensim.models.ldamodel import LdaModel
import pickle

# Data preprocessing
path1 = "../data.nosync/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
path2 = "../data.nosync/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
df_pd = pp.parquet_transform(path1, path2, n=1000)

text_preproc = pp.text_preproc_maker(list(STOP_WORDS))
df_pd['TEXT_PROCESSED'] = df_pd['TEXT'].apply(text_preproc)

data = pp.text_aggregator(df_pd, metadata='DATE', min_len=500)

dictionary, corpus = pp.gensim_prep(data)

# Data Modelling
lda = LdaModel(corpus, num_topics=10)

# pickle the model
lda_pickle = {"model": lda, "dictionary": dictionary, "corpus": corpus}
pickle.dump(lda_pickle, open('../output/lda_model.pickle', 'wb'))


