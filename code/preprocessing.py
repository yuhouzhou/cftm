from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import string


def parquet_transform(path1, path2, n=-1):
    sc = SparkContext('local')
    spark = SparkSession(sc)

    # Load parquet file into spark dataframe
    parquetFile = spark.read.parquet(
        "../data/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet")
    parquetFile.createOrReplaceTempView("parquetFile")
    parquetFile2 = spark.read.parquet(
        "../data/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet")
    parquetFile2.createOrReplaceTempView("parquetFile2")

    # Select columns which are needed
    df = spark.sql("""
    SELECT
        /* T0.KATEGORIE_2     AS CATEGORY_2,
        T0.KATEGORIE_1     AS CATEGORY_1,
        T1.ERGEBNISSATZ_ID AS RESPONSE_ID,
        T0.STIMMUNG          AS SENTIMENT,
        T1.DATUM_ID        AS DATE, */
        T1.ANTWORT_WERT    AS TEXT
    FROM
        parquetFile2 T0,
        parquetFile T1
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


def text_preproc_maker(nlp, stopwords):
    def text_preproc(sentence):
        mytokens = nlp.tokenizer(sentence)
        mytokens = [word.lemma_.lower().strip() for word in mytokens]
        mytokens = [word for word in mytokens if word not in stopwords and word not in string.punctuation]
        return mytokens

    return text_preproc


if __name__ == "__main__":
    import spacy
    from spacy.lang.de.stop_words import STOP_WORDS

    path1 = "../data/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
    path2 = "../data/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
    df_pd = parquet_transform(path1, path2, n=100)

    nlp = spacy.load('de')
    stopwords = list(STOP_WORDS)
    text_preproc = text_preproc_maker(nlp, stopwords)

    data = df_pd['TEXT'].apply(text_preproc)
    print(data)
