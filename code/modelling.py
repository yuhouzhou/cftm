from code import preprocessing as pp
import spacy
from spacy.lang.de.stop_words import STOP_WORDS

path1 = "../data/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
    path2 = "../data/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
    df_pd = pp.parquet_transform(path1, path2, n=100)

    nlp = spacy.load('de')
    stopwords = list(STOP_WORDS)
    text_preproc = pp.text_preproc_maker(nlp, stopwords)

    data = df_pd['TEXT'].apply(text_preproc)


