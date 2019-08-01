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
import datetime

parser = argparse.ArgumentParser()

parser.add_argument(
    '--preprocessing',
    type=int, nargs='?', const=1, default=1,
    help='Enter 1 to preprocess parquet files; 0 to use pickled training file'
)
parser.add_argument(
    '--observation_n',
    type=int, nargs='?', const=-1,default=-1,
)
parser.add_argument(
    '--agg_length',
    type=int, nargs='?', const=-1,default=-1
)
parser.add_argument(
    '--modelling',
    type=int, nargs='?', const=1, default=1,
    help='Enter 1 to run lda models; 0 to use pickled lda model '
)
parser.add_argument(
    '--topics_n_min', nargs='?', const=10, default=10,
    type=int
)
parser.add_argument(
    '--topics_n_max', nargs='?', const=20, default=20,
    type=int
)

args = parser.parse_args()

dt = str(datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S'))

if args.preprocessing:
    # Parsing
    path1 = "../data.nosync/customer_feedbacks/part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
    path2 = "../data.nosync/customer_feedbacks_cat/part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet"
    df_pd = cftm_parser.parquet_transform(path1, path2, n=args.observation_n)

    # Preprocessing
    stopwords = list(STOP_WORDS)
    texts, dictionary, corpus = pp.preprocessor(df_pd, stopwords=stopwords, language='de', text='TEXT', metadata='DATE',
                                                min_len=args.agg_length)
    training_data = {"texts": texts, "dictionary": dictionary, "corpus": corpus}
    pickle.dump(training_data, open('../output/training_data.pickle', 'wb'))
else:
    training_data = pickle.load(open('../output/training_data.pickle', 'rb'))
    texts, dictionary, corpus = training_data['texts'], training_data['dictionary'], training_data['corpus']

if args.modelling:
    # Model Generation
    lda_lst = []
    coherence_lst = []
    n_topics_min = args.topics_n_min
    n_topics_max = args.topics_n_max
    print("Topic Modelling starts...")
    for i in tqdm(range(n_topics_min, n_topics_max + 1)):
        # Data Modelling
        lda = LdaModel(corpus, num_topics=i, random_state=1)
        lda_lst.append(lda)

        # Data Evaluation
        coherence_model_lda = CoherenceModel(model=lda, texts=texts, corpus=corpus, dictionary=dictionary,
                                             coherence='u_mass')
        coherence = coherence_model_lda.get_coherence()
        coherence_lst.append(coherence)

    # Model Selection
    lda_pickle = {"model_lst": lda_lst, "coherence_lst": coherence_lst,
                  "n_topics_min": n_topics_min, "n_topics_max": n_topics_max}
    pickle.dump(lda_pickle, open('../output/lda_model_n_coherence_lst.pickle', 'wb'))
    pickle.dump(lda_pickle, open('../output/archive/lda_model_n_coherence_lst_'+dt+'.pickle', 'wb'))
else:
    lda_pickle = pickle.load(open('../output/lda_model_n_coherence_lst.pickle', 'rb'))
    lda_lst, coherence_lst, n_topics_min, n_topics_max = lda_pickle.values()

# Plot Topic Coherence
index = int(np.argmin(coherence_lst))
lda = lda_lst[index]
plt.scatter(range(n_topics_min, n_topics_max + 1), coherence_lst)
plt.scatter(n_topics_min + index, coherence_lst[index], color='r')
plt.annotate(str(n_topics_min + index)+', '+str(coherence_lst[index]), (n_topics_min + index, coherence_lst[index]))
plt.title('Topic Coherence vs. Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Topic Coherence (By UMass)')
plt.tight_layout()
plt.savefig('../output/coherence.png')
if args.modelling:
    plt.savefig('../output/archive/coherence_'+dt+'.png')

# Data Visualization
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary=dictionary)
pyLDAvis.save_html(vis, '../output/lda.html')
webbrowser.open(os.path.abspath('../output/lda.html'), new=2)