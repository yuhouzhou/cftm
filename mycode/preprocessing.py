import string
import spacy
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm

def text_preproc_maker(stopwords, language='de'):
    nlp = spacy.load(language)

    # TODO: increase the efficiency of text_preproc; Now: 8 min 12 sec for 1,300,724 entries of feedback
    def text_preproc(sentence):
        tokens = nlp.tokenizer(sentence)
        tokens = [token.lemma_.lower().strip() for token in tokens]
        tokens = [token for token in tokens if token not in stopwords and token not in string.punctuation]
        return tokens

    return text_preproc


def text_aggregator(df_pd, metadata=None, min_len=-1):
    if metadata is not None:
        # Speed: 151 milliseconds to concatenate 200,000 feedback to minimum length of 2000 words
        if min_len > 0:
            print("> Text aggregation started...")
            if metadata == 'DATE':
                texts = []
                tokens_agg = []
                for tokens in tqdm(df_pd['TEXT_PROCESSED']):
                    if len(tokens_agg) < min_len:
                        tokens_agg += tokens
                    else:
                        texts.append(tokens_agg)
                        tokens_agg = []
                # Append the rest of tokens to data
                if tokens_agg is not []:
                    texts.append(tokens_agg)
            if metadata == 'SENTIMENT':
                pass
        else:
            print("> Text aggregation is skipped...")
            texts = df_pd['TEXT_PROCESSED']

    return texts


def gensim_prep(texts):
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return texts, dictionary, corpus


def preprocessor(df_pd, stopwords, language='de', text='TEXT', metadata=None, min_len=300):
    text_preproc = text_preproc_maker(stopwords, language)
    print('> Text preprocessing started...')
    tqdm.pandas()
    df_pd[text + '_PROCESSED'] = df_pd[text].progress_apply(text_preproc)
    texts = text_aggregator(df_pd, metadata, min_len)
    return gensim_prep(texts)


if __name__ == "__main__":
    from spacy.lang.de.stop_words import STOP_WORDS
    import cftm_parser

    path1 = "../data.nosync/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
    path2 = "../data.nosync/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
    df_pd = cftm_parser.parquet_transform(path1, path2, n=-1)

    stopwords = list(STOP_WORDS)
    text_preproc = text_preproc_maker(stopwords)

    df_pd['TEXT_PROCESSED'] = df_pd['TEXT'].apply(text_preproc)
    texts = text_aggregator(df_pd, metadata='DATE', min_len=-1)
    for i in range(len(texts)):
        print("The length of Doc {} is {}".format(i, len(texts[i])))
