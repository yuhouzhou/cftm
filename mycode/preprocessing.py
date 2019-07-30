import string
import spacy
from gensim.corpora.dictionary import Dictionary


def text_preproc_maker(stopwords, language='de'):
    nlp = spacy.load(language)
    def text_preproc(sentence):
        mytokens = nlp.tokenizer(sentence)
        mytokens = [word.lemma_.lower().strip() for word in mytokens]
        mytokens = [word for word in mytokens if word not in stopwords and word not in string.punctuation]
        return mytokens

    return text_preproc

def text_aggregator(df_pd, metadata=None, min_len=300):
    if metadata is not None:
        # TODO: increase the efficiency of aggregation: numpy or spark
        if metadata == 'DATE':
            data = []
            tokens_agg = []
            for tokens in df_pd['TEXT_PROCESSED']:
                if len(tokens_agg) < min_len:
                    tokens_agg += tokens
                else:
                    data.append(tokens_agg)
                    tokens_agg = []
            # Append the rest of tokens to data
            if tokens_agg is not []:
                data.append(tokens_agg)
        if metadata ==  'SENTIMENT':
            pass

    return data

def gensim_prep(data):
    dictionary = Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
    return dictionary, corpus

def preprocessor(df_pd, stopwords, language='de', text = 'TEXT', metadata=None, min_len=300):
    text_preproc = text_preproc_maker(stopwords, language)
    df_pd[text+'_PROCESSED'] = df_pd[text].apply(text_preproc)
    data = text_aggregator(df_pd, metadata, min_len)
    return gensim_prep(data)

if __name__ == "__main__":
    from spacy.lang.de.stop_words import STOP_WORDS

    path1 = "../data.nosync/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
    path2 = "../data.nosync/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
    df_pd = parquet_transform(path1, path2, n=100)

    stopwords = list(STOP_WORDS)
    text_preproc = text_preproc_maker(stopwords)

    df_pd['TEXT_PROCESSED'] = df_pd['TEXT'].apply(text_preproc)
    data = text_aggregator(df_pd, metadata='DATE', min_len=300)
    for i in range(len(data)):
        print("The length of Doc {} is {}".format(i, len(data[i])))
