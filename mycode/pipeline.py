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
    'pipeline',
    type=int, nargs=3, default=[1, 1, 1],
    help=('Define pipeline: \n'
          '    Require 3 numbers. Each number switches on or off a part of the pipeline.\n'
          '    The first position is preprocessing,\n'
          '    the second position is modelling,\n'
          '    the third position is visualization.\n'
          '    Enter 1 to turn on, 0 to turn off the specific part of pipeline.\n'
          '    Example: 1 1 1 to run the full pipeline,\n'
          '    1 1 0 to run preprocessing and modelling but without visualization.\n'
          '    ')
)

parser.add_argument(
    '-o', '--observation_n',
    type=int, nargs='?', const=-1, default=-1,
)
parser.add_argument(
    '-a', '--agg_length',
    type=int, nargs='?', const=-1, default=-1
)

parser.add_argument(
    '-r', '--topic_range',
    type=int, nargs=2, default=[10, 15],

)

args = parser.parse_args()

parquet_path1 = "../data.nosync/customer_feedbacks/" \
                "part-00000-985ad763-a6d6-4ead-a6dd-c02279e9eeba-c000.snappy.parquet"
parquet_path2 = '../data.nosync/customer_feedbacks_cat/' \
                'part-00000-4820af87-4b19-4958-a7a6-7ed03b76f1b1-c000.snappy.parquet'
data_path = '../output/training_data.pickle'
model_path = '../output/lda_model_n_coherence_lst.pickle'
pic_path = '../output/coherence.png'
html_path = '../output/lda.html'
dt = str(datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S'))

if args.pipeline[0]:
    # Parsing
    df_pd = cftm_parser.parquet_transform(parquet_path1, parquet_path2, n=args.observation_n)

    # Preprocessing
    stopwords = list(STOP_WORDS)
    texts, dictionary, corpus = pp.preprocessor(df_pd, stopwords=stopwords, language='de', text='TEXT', metadata='DATE',
                                                min_len=args.agg_length)
    training_data = {"texts": texts, "dictionary": dictionary, "corpus": corpus}
    pickle.dump(training_data, open(data_path, 'wb'))
elif args.pipeline[1] or args.pipeline[2]:
    try:
        training_data = pickle.load(open(data_path, 'rb'))
        texts, dictionary, corpus = training_data['texts'], training_data['dictionary'], training_data['corpus']
    except FileNotFoundError:
        print("Training Data Not Found!")
        exit(1)

if args.pipeline[1]:
    # Model Generation
    lda_lst = []
    coherence_lst = []
    n_topic_min = args.topic_range[0]
    n_topics_max = args.topic_range[1]
    print("Topic Modelling starts...")
    for i in tqdm(range(n_topic_min, n_topics_max + 1)):
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
                  "n_topics_min": n_topic_min, "n_topics_max": n_topics_max}
    pickle.dump(lda_pickle, open(model_path, 'wb'))
    pickle.dump(lda_pickle, open(os.path.splitext(model_path) + dt + '.pickle', 'wb'))
elif args.pipeline[2]:
    try:
        lda_pickle = pickle.load(open(model_path, 'rb'))
        lda_lst, coherence_lst, n_topic_min, n_topics_max = lda_pickle.values()
    except FileNotFoundError:
        print("Model Not Found!")
        exit(1)

if args.pipeline[2]:
    # Plot Topic Coherence
    index = int(np.argmin(coherence_lst))
    lda = lda_lst[index]
    plt.scatter(range(n_topic_min, n_topics_max + 1), coherence_lst)
    plt.scatter(n_topic_min + index, coherence_lst[index], color='r')
    plt.annotate(str(n_topic_min + index) + ', ' + str(coherence_lst[index]),
                 (n_topic_min + index, coherence_lst[index]))
    plt.title('Topic Coherence vs. Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Topic Coherence (By UMass)')
    plt.tight_layout()
    plt.savefig(pic_path)
    if args.pipeline[1]:
        plt.savefig(os.path.splitext(pic_path) + dt + '.png')

    # Data Visualization
    vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary=dictionary)
    pyLDAvis.save_html(vis, html_path)
    webbrowser.open(os.path.abspath(html_path), new=2)