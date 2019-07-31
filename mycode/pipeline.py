import cftm_parser
import preprocessing as pp
from spacy.lang.de.stop_words import STOP_WORDS
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pickle
import pyLDAvis.gensim
import webbrowser
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# TODO: Before deployment finish arg parser; enclose the pipeline into one function
# n_topic_min and max, etc....
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--preprocessing',
        type = int,
        help ='Use preprocessing'
    )
    parser.add_argument(
        '--modelling',
        type=int,
        help='Use modelling'
    )
    args = parser.parse_args()


if args.preprocessing:
    # Parsing
    path1 = "../data.nosync/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
    path2 = "../data.nosync/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
    df_pd = cftm_parser.parquet_transform(path1, path2, n=20000)

    # Preprocessing
    stopwords = list(STOP_WORDS)
    texts, dictionary, corpus = pp.preprocessor(df_pd, stopwords=stopwords, language='de', text='TEXT', metadata='DATE',
                                                min_len=-1)
    training_data = {"texts": texts, "dictionary": dictionary, "corpus": corpus}
    pickle.dump(training_data, open('../output/training_data.pickle', 'wb'))
else:
    training_data = pickle.load(open('../output/training_data.pickle', 'rb'))
    texts, dictionary, corpus = training_data['texts'], training_data['dictionary'], training_data['corpus']

if args.modelling:
    # Model Generation
    lda_lst = []
    coherence_lst = []
    n_topics_min = 95
    n_topics_max = 100
    for i in tqdm(range(n_topics_min, n_topics_max + 1)):
        # Data Modelling
        lda = LdaModel(corpus, num_topics=i)
        lda_lst.append(lda)

        # Data Evaluation
        coherence_model_lda = CoherenceModel(model=lda, texts=texts, corpus=corpus, dictionary=dictionary,
                                             coherence='u_mass')
        coherence = coherence_model_lda.get_coherence()
        coherence_lst.append(coherence)

    # Model Selection
    index = np.argmin(coherence_lst)
    lda = lda_lst[index]
    lda_pickle = {"model": lda, "coherence_lst":coherence_lst}
    pickle.dump(lda_pickle, open('../output/lda_model_n_coherence_lst.pydict', 'wb'))
else:
    lda_pickle = pickle.load(open('../output/lda_model_n_coherence_lst.pydict', 'rb'))
    lda = lda_pickle['model']
    coherence_lst = lda_pickle['coherence_lst']

# Plot Topic Coherence
    plt.scatter(range(n_topics_min, n_topics_max + 1), coherence_lst)
    plt.scatter(n_topics_min + index, coherence_lst[index], color='r')
    plt.title('Topic Coherence vs. Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Topic Coherence')
    plt.savefig('../output/coherence.png')

# Data Visualization
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary=dictionary)
pyLDAvis.save_html(vis, '../output/lda.html')
webbrowser.open(os.path.abspath('../output/lda.html'), new=2)



