# -*- coding: utf-8 -*-
import sys
sys.path.append("../libs")
import vectortools, simpletools, config
import scipy.cluster.hierarchy as hcluster
import numpy as np
import os, pickle
from collections import Counter


with open(config.RU_MODEL_PATH, 'rb') as f:
    RU_MODEL = pickle.load(f)

with open(config.EN_MODEL_PATH, 'rb') as f:
    EN_MODEL = pickle.load(f)

models = {
    'ru': RU_MODEL,
    'en': EN_MODEL
}


def cmd_top(src_dir):
    # prepare samples
    joblist = list(simpletools.collect_files(src_dir))

    # process with multiprocessing
    categoriesToVecs, vecToTitlesFiles = top_merge(simpletools.parallel_processor(top_worker, joblist, split=True))

    # prepare result
    result = []
    clusterInfo = []
    for category in categoriesToVecs:
        X = categoriesToVecs[category]
        if X.shape[0] > 1:
            clusters = hcluster.fclusterdata(X, config.CLUSTERIZATION_THRESHOLD, criterion="distance", metric="cosine")
            counts = Counter(clusters)
            for cluster in counts:
                info = {
                    'category': config.invertedClasses[category],
                    'title': "",
                    'articles': []
                }
                indexes = [i for i, e in enumerate(clusters) if e == cluster]
                vectors = np.take(X, indexes, axis=0)
                info["title"] = vecToTitlesFiles[tuple(vectors[0])]["title"]
                info["articles"] = [vecToTitlesFiles[tuple(vector)]["filename"] for vector in vectors]
                clusterInfo.append(info)

    result.append({
        "category": "any",
        "threads": sorted(clusterInfo, key=lambda i: len(i['articles']), reverse=True)
    })
    for category in config.classes:
        threads = list(filter(lambda x: x["category"] == category, clusterInfo))
        threads = [{'title': t['title'], 'articles': t['articles']} for t in threads]
        result.append({
            "category": category,
            "threads": sorted(threads, key=lambda i: len(i['articles']), reverse=True)
        })
    return result


def top_worker(files):
    categoriesToVecs = {
        0: np.empty((0, config.EMBEDDINGS_DIM), int),
        1: np.empty((0, config.EMBEDDINGS_DIM), int),
        2: np.empty((0, config.EMBEDDINGS_DIM), int),
        3: np.empty((0, config.EMBEDDINGS_DIM), int),
        4: np.empty((0, config.EMBEDDINGS_DIM), int),
        5: np.empty((0, config.EMBEDDINGS_DIM), int),
        6: np.empty((0, config.EMBEDDINGS_DIM), int)
    }
    vecToTitlesFiles = {}
    for filepath in files:
        page = simpletools.parsePage(filepath)
        title = page["title"]
        article = page["article"]
        text = article + title
        lang = simpletools.detectLang(article)
        if lang and simpletools.is_news_title(lang, title):
            vectorClass, lang = vectortools.textToVector(page["article"])
            vectorCluster, lang = vectortools.textToVectorClusterization(text)
            if vectorClass.size != 0:
                filename = os.path.basename(filepath)
                y = models[lang].predict(vectorClass.reshape((1, 300)))[0]
                categoriesToVecs[y] = np.vstack([categoriesToVecs[y], vectorCluster])
                vecToTitlesFiles[tuple(vectorCluster)] = {
                    "title": page["title"],
                    "filename": filename
                }
    return categoriesToVecs, vecToTitlesFiles


def top_merge(data):
    vecToTitlesFiles = {}
    categoriesToVecs = {
        0: np.empty((0, config.EMBEDDINGS_DIM), int),
        1: np.empty((0, config.EMBEDDINGS_DIM), int),
        2: np.empty((0, config.EMBEDDINGS_DIM), int),
        3: np.empty((0, config.EMBEDDINGS_DIM), int),
        4: np.empty((0, config.EMBEDDINGS_DIM), int),
        5: np.empty((0, config.EMBEDDINGS_DIM), int),
        6: np.empty((0, config.EMBEDDINGS_DIM), int)
    }
    for cats, d in data:
        for cat in cats:
            categoriesToVecs[cat] = np.vstack([categoriesToVecs[cat], cats[cat]])
        vecToTitlesFiles.update(d)
    return categoriesToVecs, vecToTitlesFiles