# -*- coding: utf-8 -*-
import sys
sys.path.append("../libs")
import vectortools, simpletools, config
import scipy.cluster.hierarchy as hcluster
import numpy as np
import os


def cmd_threads(src_dir):
    # prepare samples
    joblist = list(simpletools.collect_files(src_dir))

    # process with multiprocessing
    X, vecToTitlesFiles = threads_merge(simpletools.parallel_processor(threads_worker, joblist, split=True))
    clusters = hcluster.fclusterdata(X, config.CLUSTERIZATION_THRESHOLD, criterion="distance", metric="cosine")

    # prepare result
    result = []
    groupedClasters = sorted(enumerate(clusters), key=lambda tup: tup[1])
    currentCluster = 0
    currentGroup = {}
    for i, cluster in groupedClasters:
        vec = tuple(X[i])
        title = vecToTitlesFiles[vec]["title"]
        filename = vecToTitlesFiles[vec]["filename"]
        if cluster != currentCluster:
            if currentGroup:
                result.append(currentGroup)
            currentGroup = {
                "title": title,
                "articles": [filename]
            }
        else:
            currentGroup["articles"].append(filename)
        currentCluster = cluster
    if currentGroup:
        result.append(currentGroup)

    return result


def threads_merge(data):
    vecToTitlesFiles = {}
    X = np.empty((0, config.EMBEDDINGS_DIM), int)
    for x, d in data:
        X = np.vstack([X, x])
        vecToTitlesFiles.update(d)
    return X, vecToTitlesFiles


def threads_worker(files):
    X = np.empty((0, config.EMBEDDINGS_DIM), int)
    vecToTitlesFiles = {}
    for filepath in files:
        page = simpletools.parsePage(filepath)
        title = page["title"]
        article = page["article"]
        text = article + title
        lang = simpletools.detectLang(article)
        if lang and simpletools.is_news_title(lang, title):
            vector, lang = vectortools.textToVectorClusterization(text)
            if lang and vector.size != 0:
                X = np.vstack([X, vector])
                filename = os.path.basename(filepath)
                vecToTitlesFiles[tuple(vector)] = {
                    "title": page["title"],
                    "filename": filename
                }
    return X, vecToTitlesFiles