# Customer Feedback Topic Extraction

## Data Description

Data is from two Apache parquet files of the overall size 1.47 GB.  The parquet file in `customer_feedbacks` contains 101 columns, and the one in `customer_feedbacks_cat` contains 7 columns. The content of customer feedback is stored in the column `ANTWORT_WERT`, There are 1,917,490 feedbacks including duplicates since 14/04/2012.The sentiment of the feedback is stored in the column `STIMMUNG`.

## Project Pipeline

### Data Importing

* **Spark**
  Use `spark.read.parquet()` function to read parquet files, which preserves the schema of the original data. `spark.sql()` returns a spark dataframe. We can write query to select the data we are interested in.

### Data Parsing

By using `spark.sql()` we get the columns of interests, such as "Date", "Sentiment", "Text", etc.

### Data preprocessing

* Drop duplicates (transform the spark dataframe to pandas dataframe)
* Aggregate short feedback to long documents by specified aggregation strategy
* preprocessing pipeline
  * Tokenization
  * lemmatization
  * Lowercasing
  * Striping white spaces
  * removing stop words and punctuations
* Call gensim methods to transform string texts to gensim dictionary, and gensim corpus. 

### Data Modeling and Selection

* Use gensim `LdaModel` in distributed mode to train models. Generate a user-specified number of models. 
* Evaluation these models by Topic Coherence, and choose the one with the best evaluation result.

### Data Visualization

Use `matplotlib` to visualize the topic coherence change of models.

Use `pyLDAvis.sklearn` to visualize the topic result.

