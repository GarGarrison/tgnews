# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../libs")
import os, pickle
import simpletools, vectortools, config


with open(config.RU_MODEL_PATH, 'rb') as f:
    RU_MODEL = pickle.load(f)

with open(config.EN_MODEL_PATH, 'rb') as f:
    EN_MODEL = pickle.load(f)

models = {
    'ru': RU_MODEL,
    'en': EN_MODEL
}


def cmd_categories(src_dir):
    # prepare samples
    joblist = simpletools.collect_files(src_dir)

    # process with multiprocessing
    data = simpletools.parallel_processor(categories_worker, joblist)

    article_by_categories = {cat: [] for cat in config.classes.keys()}
    for i in data:
        cat_id, filename = i[0], i[1]
        category = config.invertedClasses.get(cat_id, None)
        if filename and category:
            article_by_categories[category].append(filename)

    return [{"category": cat, "articles": articles} for cat, articles in article_by_categories.items()]


def categories_worker(filepath):
    page = simpletools.parsePage(filepath)
    title = page["title"]
    article = page["article"]
    lang = simpletools.detectLang(article)
    if lang and simpletools.is_news_title(lang, title):
        vector, lang = vectortools.textToVector(article)
        if lang and vector.size != 0:
            filename = os.path.basename(filepath)
            y = models[lang].predict(vector.reshape((1, 300)))[0]
            return y, filename
    return None, None