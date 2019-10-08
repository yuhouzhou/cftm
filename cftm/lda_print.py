import pickle
import numpy as np
from pprint import pprint

file_path = '../output/lda_model_n_coherence_lst.pickle'
lda_pickle= pickle.load(open(file_path, 'rb'))
lda_lst, coherence_lst, n_topic_min, n_topics_max, step = lda_pickle.values()
index = int(np.argmax(coherence_lst))
lda = lda_lst[index]
pprint(lda.print_topics())