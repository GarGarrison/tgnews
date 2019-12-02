# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../libs")
from multiprocessing import cpu_count
from os.path import join
import nltk

# because HyperThreading
PROCESSES_COUNT = 2*cpu_count()

FIXED_LANGUAGES = ('en', 'ru')

# gar
classes = {
    'economy': 0,
    'entertainment': 1,
    'science': 2,
    'society': 3,
    'sports': 4,
    'technology': 5,
    'other': 6
}
invertedClasses = {v:k for k,v in classes.items()}
EMBEDDINGS_DIM = 300
CATEGORIES_DIM = len(classes)
KEYWORDS_COUNT = 30

CLUSTERIZATION_KEYWORDS_COUNT = 14
CLUSTERIZATION_THRESHOLD = 0.16


DATA_DIR = "data"

nltk.data.path.append("nltk_data")

RU_SPACY_MODEL_PATH = join(DATA_DIR, 'spacy', "ru2")
EN_SPACY_MODEL_PATH = join(DATA_DIR, 'spacy', "en_core_web_sm")

RU_MODEL_PATH = join(DATA_DIR, "ru_logreg2.pickle")
EN_MODEL_PATH = join(DATA_DIR, "en_logreg2.pickle")

RU_VEC_DICT_PATH = join(DATA_DIR, "ru_vector_dict2.pickle")
EN_VEC_DICT_PATH = join(DATA_DIR, "en_vector_dict2.pickle")

RU_TFIDF_PATH = join(DATA_DIR, "ru_tfidf2.pickle")
EN_TFIDF_PATH = join(DATA_DIR, "en_tfidf2.pickle")
