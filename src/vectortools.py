# -*- coding: utf-8 -*-
import sys
sys.path.append("../libs")
import pickle
import simpletools, config
import numpy as np

with open(config.RU_VEC_DICT_PATH, 'rb') as f:
    RU_VECTORS = pickle.load(f)

with open(config.EN_VEC_DICT_PATH, 'rb') as f:
    EN_VECTORS = pickle.load(f)

with open(config.RU_TFIDF_PATH, 'rb') as f:
    RU_TFIDF = pickle.load(f)

with open(config.EN_TFIDF_PATH, 'rb') as f:
    EN_TFIDF = pickle.load(f)

wordToVecDict = {
    'ru': RU_VECTORS,
    'en': EN_VECTORS
}

tfidfDict = {
    'ru': RU_TFIDF,
    'en': EN_TFIDF
}

featureWordsDict = {
    'ru': RU_TFIDF.get_feature_names(),
    'en': EN_TFIDF.get_feature_names()
}

def keyWordsToVector(words, wordToVec, EMBEDDINGS_DIM=config.EMBEDDINGS_DIM):
    vectors = np.empty((0, EMBEDDINGS_DIM),int)
    for w in words:
        vectors = np.vstack([vectors, wordToVec[w]])
    return np.sum(vectors, axis=0)

def keyWordsToVectorMedian(words, wordToVec, EMBEDDINGS_DIM=config.EMBEDDINGS_DIM):
    vectors = np.empty((0, EMBEDDINGS_DIM),int)
    for w in words:
        vectors = np.vstack([vectors, wordToVec[w]])
    return np.median(vectors, axis=0)

# def keyWordsToSentence(words, wordToVec, EMBEDDINGS_DIM=config.EMBEDDINGS_DIM):
#     sentence_len = len(words)
#     vectors = np.zeros((sentence_len, EMBEDDINGS_DIM),int)
#     for i in range(0,len(words)):
#         vectors[i] = wordToVec[words[i]]
#     return vectors

def getKeyWords(matrix, feature_names, kw_count=config.KEYWORDS_COUNT):
    words = []
    for col, _ in simpletools.sortKeyWords(matrix.tocoo())[:kw_count]:
        words.append(feature_names[col])
    return words

def textToKeyWords(text, kw_count=config.KEYWORDS_COUNT):
    lang = simpletools.detectLang(text)
    # if lang:
    tfidf = tfidfDict[lang]
    feature_names = featureWordsDict[lang]
    text = simpletools.tokenizeText(text, lang=lang)
    text_matrix = tfidf.transform([text])
    keywords = getKeyWords(text_matrix, feature_names, kw_count)
    return keywords, lang
    # return None,None

def textToVector(text, kw_count=config.KEYWORDS_COUNT):
    keywords, lang = textToKeyWords(text, kw_count=kw_count)
    if keywords:
        vectors = wordToVecDict[lang]
        return keyWordsToVector(keywords, vectors), lang
    return np.array([]), lang

def textToVectorClusterization(text, kw_count=config.CLUSTERIZATION_KEYWORDS_COUNT):
    keywords, lang = textToKeyWords(text, kw_count=kw_count)
    if keywords:
        vectors = wordToVecDict[lang]
        return keyWordsToVectorMedian(keywords, vectors), lang
    return np.array([]), lang

def pageToKeyWords(filepath, kw_count=config.KEYWORDS_COUNT):
    text = simpletools.parsePage(filepath)["article"]
    return textToKeyWords(text, kw_count=kw_count)

# def pageToSentence(filepath, kw_count=config.KEYWORDS_COUNT):
#     keywords, lang = pageToKeyWords(filepath, kw_count)
#     if keywords:
#         vectors = wordToVecDict[lang]
#         return keyWordsToSentence(keywords, vectors)
#     return np.array([])

#def pageToVector(filepath, kw_count=config.KEYWORDS_COUNT):
#    keywords, lang = pageToKeyWords(filepath, kw_count)
#    if keywords:
#        vectors = wordToVecDict[lang]
#        return keyWordsToVector(keywords, vectors), lang
#    return np.array([]), lang
