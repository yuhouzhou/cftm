# Customer Feedback Topic Extraction

## Data Inspection

Data is from two Apache parquet files of the overall size 1.47 GB.  The parquet file in `customer_feedbacks` contains 101 columns, and the one in `customer_feedbacks_cat` contains 7 columns. The content of customer feedback is stored in the column `ANTWORT_WERT`, There are 1,917,490 feedbacks including duplicates since 14/04/2012.The sentiment of the feedback is stored in the column `STIMMUNG`

## Data Importing

* Pandas (later using incremental learning methods to train models)
  * Directly using Python Pandas read_parquet leads to error "Out of  Memory"
  * [“Large data" work flows using pandas](https://stackoverflow.com/questions/14262433/large-data-work-flows-using-pandas)
* Dask
* postgreSQL
* **Spark**
  Use `spark.read.parquet()` function to read parquet files, which preserves the schema of the original data. `spark.sql` returns a spark dataframe. We can write query to select the data we are interested in.
* Command-line tools with streaming methods
  *  [Command-line Tools can be 235x Faster than your Hadoop Cluster](https://adamdrake.com/command-line-tools-can-be-235x-faster-than-your-hadoop-cluster.html)

**References:**

[Python/Pandas 如何处理百亿行，数十列的数据？](Python/Pandas 如何处理百亿行，数十列的数据？)

## Data preparation

* Drop duplicates (transform the spark dataframe to pandas dataframe)
* Create preprocessing function
  * Apply standard nlp process to every feedback. Tokenize, part-of-speech tagging, parsing and named entity recognition. 
  * Change words to lower cases; remove stopwords; change to string type for scikitlearn

## Data Modeling

* Use `ContVectorizer` in `scikitlearn` to vectorize text data.
* Set the number of clusters
* Train the model

## Data Visualization

Use `pyLDAvis.sklearn` to visualize the result

