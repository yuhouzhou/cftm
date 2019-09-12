refrom pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


def parquet_transform(path1, path2, n=-1):
    """Transform the project parquet file to Pandas Dataframe

    Args:
        path1: the path of the first parquet file
        path2: the path of the second parquet file
        n: the number of observations will be included in the return

    Returns: a Pandas Dataframe including text and its metadata

    """

    sc = SparkContext('local')
    spark = SparkSession(sc)

    print("> File loading started...")
    # Load parquet file into spark dataframe
    parquetFile1 = spark.read.parquet(path1)
    parquetFile1.createOrReplaceTempView("parquetFile1")
    parquetFile2 = spark.read.parquet(path2)
    parquetFile2.createOrReplaceTempView("parquetFile2")
    print("> File loading finshied...")

    print("> Parsing started...")
    # Select columns which are needed
    query_p1 = """
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
    """
    if n >= 0:
        query_p2 = " LIMIT " + str(n)
    else:
        query_p2 = ""
    query = query_p1 + query_p2

    df = spark.sql(query)
    print("> Parsing finished!")

    # Convert Spark dataframe to Pandas dataframe
    print("> Dropping duplicates and transforming to Pandas dataframe started...")
    df_pd = df.dropDuplicates().toPandas()
    print("> Dropping duplicates and transforming to Pandas dataframe finished!")

    sc.stop()

    return df_pd


if __name__ == "__main__":
    path1 = "../data.nosync/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
    path2 = "../data.nosync/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
    df_pd = parquet_transform(path1, path2, n=100)
    print(df_pd)
