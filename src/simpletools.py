# -*- coding: utf-8 -*-
# import sys
# sys.path.append("../src")
# sys.path.append("../libs")
import os
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
from selectolax.parser import HTMLParser
from multiprocessing import Pool
import config
from spacy.util import set_data_path
import spacy
# todo: change on prod!
#from whatthelang import WhatTheLang
from langdetect import detect, DetectorFactory
from langdetect.detector_factory import init_factory

DetectorFactory.seed = 0

# load models
#set_data_path(SPACY_MODEL_PATH)
nlpEn = spacy.load(config.RU_SPACY_MODEL_PATH)
nlpRu = spacy.load(config.RU_SPACY_MODEL_PATH)

#wtl = WhatTheLang()
ru_normalizer = MorphAnalyzer()
en_normalizer = WordNetLemmatizer()

ru_stopwords = {'', 'такой', 'быть', 'нас', 'больше', 'еще', 'на', 'его', 'один', 'тоже', 'а', 'или', 'какой', 'этом', 'это', 'через', 'но', 'другой', 'куда', 'чуть', 'нет', 'хоть', 'чтоб', 'с', 'по', 'можно', 'о', 'будет', 'он', 'будто', 'того', 'чтобы', 'же', 'у', 'ты', 'к', 'когда', 'почти', 'нельзя', 'ли', 'сам', 'для', 'нее', 'него', 'наконец', 'ей', 'всю', 'во', 'всегда', 'тем', 'не', 'если', 'вы', 'этот', 'свою', 'себя', 'над', 'тот', 'хорошо', 'уже', 'и', 'об', 'меня', 'после', 'ним', 'чем', 'три', 'раз', 'ней', 'теперь', 'том', 'мне', 'совсем', 'при', 'ему', 'иногда', 'уж', 'здесь', 'даже', 'всего', 'они', 'от', 'тебя', 'два', 'под', 'зачем', 'никогда', 'как', 'есть', 'была', 'вдруг', 'вам', 'до', 'тогда', 'им', 'ничего', 'за', 'себе', 'сейчас', 'между', 'вот', 'моя', 'она', 'лучше', 'тут', 'потом', 'разве', 'был', 'что', 'все', 'из', 'перед', 'без', 'них', 'чего', 'про', 'конечно', 'ж', 'эти', 'я', 'да', 'со', 'впрочем', 'были', 'ну', 'там', 'их', 'кто', 'мой', 'бы', 'какая', 'было', 'потому', 'только', 'надо', 'более', 'то', 'может', 'ведь', 'этой', 'эту', 'этого', 'нибудь', 'много', 'так', 'в', 'ни', 'мы', 'всех', 'ее', 'опять', 'где', 'вас'}
en_stopwords = {'', 'between', 'above', 'i', 'she', 'itself', 'any', 'both', 'ma', 'aren', "that'll", 'very', 'll', 'all', 'into', 't', 'which', 'me', 'through', 'myself', 'only', 'd', 'is', 'mustn', "it's", 'were', 'off', 'him', 'theirs', "wouldn't", 'once', "weren't", 'these', "you've", 'whom', 'did', 'just', 'didn', 'that', 'do', 'until', 'how', 'no', 'doesn', 'an', 'while', 'your', 'they', 'then', 'o', 'the', 'her', 'before', "doesn't", 'such', 'having', 'again', 'same', 'under', 'yourselves', 'after', 'had', 'y', 'my', 'as', "shouldn't", 'who', 'ain', 'this', 'below', 'wouldn', 'yourself', 'about', 'isn', "you'd", 'too', 'will', 'was', 'shan', 'wasn', 'down', 'been', 'does', 'haven', 'mightn', 've', 'up', 'its', 'them', 'yours', 'because', 'nor', 'shouldn', 'some', 'over', 's', 'to', 'out', "you're", 'against', 'why', 'ours', 'his', 'am', 'other', 'hers', "haven't", 'each', "hadn't", 'those', 'have', 'not', "should've", 'so', "shan't", 'if', 'for', 'won', "she's", 'own', 'hasn', 'with', 'at', "don't", 'ourselves', "won't", 'what', 'further', 'from', 'but', 'herself', "isn't", 'we', 'be', 'you', 'there', 'when', 'now', "aren't", 'it', 're', 'has', 'can', 'our', "needn't", 'in', 'weren', 'their', "you'll", "didn't", "hasn't", "wasn't", 'than', 'and', 'here', "mustn't", 'themselves', 'needn', 'couldn', 'few', 'or', 'of', "couldn't", 'more', 'being', "mightn't", 'm', 'most', 'he', 'are', 'by', 'hadn', 'don', 'doing', 'should', 'on', 'where', 'a', 'during', 'himself'}


def normalizeRu(word):
    return ru_normalizer.parse(word)[0].normal_form


def normalizeEn(word):
    word = word.lower()
    word = en_normalizer.lemmatize(word, pos=("a"))
    word = en_normalizer.lemmatize(word, pos="n")
    word = en_normalizer.lemmatize(word, pos="v")
    return word

normalizerDict = {
    'ru': normalizeRu,
    'en': normalizeEn
}
stopwordsDict = {
    'ru': ru_stopwords,
    'en': en_stopwords
}


def collect_files(src_path):
    return (os.path.join(sd, file) for sd, _, files in os.walk(src_path) for file in files)


def is_news_title(lang, title):
    """
        Check if html is news article.
        Returns filename if yes, None otherwise
    """
    if not title:
        return False

    if lang == 'ru':
        # normalize title
        # remove quotes to avoid problems with subject and predictive detection
        normalized_title = re.sub(r"-", '', title.lower())
        normalized_title = re.sub(r"[\"«“]\w+\s+.*?[\"»”]\.?\s?", '', normalized_title)
        normalized_title = re.sub(r"^[\w\s]*:\s?", '', normalized_title)
        # remove question sentences
        if '?' in normalized_title:
            normalized_title = ". ".join(s for s in normalized_title.split('. ') if not s.endswith('?'))

        # parse sentence
        for tok in nlpRu(normalized_title):
            if tok.dep_ == 'ROOT':
                if tok.pos_ in ['AUX', 'VERB']:
                    return True
                if tok.pos_ == 'ADJ' and 'ADJ__Degree=Pos' in tok.tag_:
                    return True
        return False
    elif lang == 'en':
        # remove endings like "bla bla bla | news.com"
        normalized_title = re.sub(r"[\||\:]\s*\w+\s+\w+\s*$", '', title.lower())
        # remove quotes authors like "somebody: bla bla bla"
        normalized_title = re.sub(r"^[\w\s]*:\s?", '', normalized_title)
        # remove quotes
        normalized_title = re.sub(r"[\"«“]\w+\s+.*?[\"»”]\.?\s?", '', normalized_title)

        for tok in nlpEn(normalized_title):
            if tok.dep_ == 'ROOT':
                if tok.pos_ in ['AUX', 'VERB']:
                    return True
                if tok.pos_ == 'ADJ' and 'ADJ__Degree=Pos' in tok.tag_:
                    return True
    return False


def detectNews(text):
    import random
    return random.randint(0, 1)


def splitList(l, chunk=4):
    split_l = []
    llen = len(l)
    last = 0
    end = int(llen/chunk) + 1
    if llen % chunk == 0:
        end = int(llen/chunk)
    for i in range(0,len(l), end):
        split_l.append(l[last:i])
        last = i
    split_l.append(l[last:])
    return split_l[1:]


def detectLang(text, detect_all=False):
    if text:
        # todo: decomment on prod
        #lang = wtl.predict_lang(text[:100])

        lang = detect(text)
        if detect_all:
            return lang
        else:
            if lang in config.FIXED_LANGUAGES:
                return lang
    return None


def getCategoryFromPath(filename):
    dirname = os.path.dirname(filename)
    return os.path.basename(dirname)


def parsePage(filepath):
    # read file
    with open(filepath, 'rb') as f:
        page_raw = f.read()

    tree = HTMLParser(page_raw)
    data = {}

    # get article
    article = tree.css_first('article')
    if article:
        data["article"] = re.sub(r'\s+', ' ', article.text().strip())
    else:
        data["article"] = ""

    # get title
    title = tree.css_first("meta[property=\"og:title\"]")
    if title:
        data['title'] = title.attributes['content']
    else:
        data["title"] = ""

    return data


def normalizeWord(word, lang):
    return normalizerDict[lang](word)


def punctuationReplace(text):
    # replace all non-alphanumeric characters and digits
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ]+", " ", text)
    return text

def widerPunctuationReplace(text):
    # replace all non-alphanumeric characters and digits
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9$%]+", " ", text)
    text = text.replace("$", " $").replace("%", " %")
    return text

def sortKeyWords(m):
    tuples = zip(m.col, m.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def tfidf_tokenizer(text):
    return text

def tokenizeText(text, lang=None):
    if not lang:
        lang = detectLang(text)
    newtext = []
    if lang:
        stopwords = stopwordsDict[lang]
        # text = punctuationReplace(text)
        text = widerPunctuationReplace(text)
        text = text.split(' ')
        for word in text:
            if word not in stopwords:
                word = normalizeWord(word, lang)
                newtext.append(word)
    del(text)
    return newtext

def classToVectors(classes):
    # returns map of y class to y vector: society:[0,0,1,0,0,0]
    y_dict = {}
    CATEGORIES_DIM = len(classes)
    for cl in classes:
        y = np.zeros(CATEGORIES_DIM)
        y[classes[cl]] = 1
        y_dict[cl] = y
    return y_dict


def parallel_processor(func, jobs, cores=config.PROCESSES_COUNT, split=False):
    # split jobs into batches, because merging PROCESSES_COUNT vectors on 1 core is faster than len(jobs) vectors
    if split:
        jobs = splitList(jobs, cores)
    with Pool(processes=cores, initializer=init_factory) as pool:
        data = pool.map(func, jobs)
        pool.close()
        pool.join()
    return data
