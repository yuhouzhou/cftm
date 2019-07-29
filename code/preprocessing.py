from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import string
import spacy
from gensim.corpora.dictionary import Dictionary

def parquet_transform(path1, path2, n=-1):
    sc = SparkContext('local')
    spark = SparkSession(sc)

    # Load parquet file into spark dataframe
    parquetFile1 = spark.read.parquet(path1)
    parquetFile1.createOrReplaceTempView("parquetFile1")
    parquetFile2 = spark.read.parquet(path2)
    parquetFile2.createOrReplaceTempView("parquetFile2")

    # Select columns which are needed
    df = spark.sql("""
    SELECT
        /* T0.KATEGORIE_2     AS CATEGORY_2,
        T0.KATEGORIE_1     AS CATEGORY_1,
        T1.ERGEBNISSATZ_ID AS RESPONSE_ID, */
        T0.STIMMUNG          AS SENTIMENT,
        T1.DATUM_ID        AS DATE,
        T1.ANTWORT_WERT    AS TEXT
    FROM
        parquetFile2 T0,
        parquetFile1 T1
    WHERE
        T0.KATEGORIE_1_ID = T1.KATEGORIE_1_ID
        AND T0.KATEGORIE_2_ID = T1.KATEGORIE_2_ID
        AND T0.STIMMUNG_ID = T1.STIMMUNG_ID
        AND (NOT T1.ANTWORT_WERT IS NULL
            AND (T1.UMFRAGE_KATEGORIE_ID = 1
                AND (T1.GRUPPE_ID = 170
                    OR T1.GRUPPE_ID = 171)))
    """)

    # Convert Spark dataframe to Pandas dataframe
    if n>=0:
        df_pd = df.dropDuplicates().toPandas()[:n]
    else:
        df_pd = df.dropDuplicates().toPandas()[:n]

    sc.stop()

    return df_pd


def text_preproc_maker(stopwords, language='de'):
    nlp = spacy.load(language)
    def text_preproc(sentence):
        mytokens = nlp.tokenizer(sentence)
        mytokens = [word.lemma_.lower().strip() for word in mytokens]
        mytokens = [word for word in mytokens if word not in stopwords and word not in string.punctuation]
        return mytokens

    return text_preproc

def text_aggregator(df_pd, metadata=None, min_len=2000):
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

if __name__ == "__main__":
    from spacy.lang.de.stop_words import STOP_WORDS

    path1 = "../data/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
    path2 = "../data/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
    df_pd = parquet_transform(path1, path2, n=100)

    stopwords = list(STOP_WORDS)
    text_preproc = text_preproc_maker(stopwords)

    df_pd['TEXT_PROCESSED'] = df_pd['TEXT'].apply(text_preproc)
    data = text_aggregator(df_pd, metadata='DATE', min_len=300)
    for i in range(len(data)):
        print("The length of Doc {} is {}".format(i, len(data[i])))
