# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../libs")
from selectolax.parser import HTMLParser
from langdetect import detect, DetectorFactory
from langdetect.detector_factory import init_factory
from config import PROCESSES_COUNT
import multiprocessing as mp
import os
import simpletools


DetectorFactory.seed = 0


def cmd_news(src_dir):
    # prepare samples
    joblist = simpletools.collect_files(src_dir)

    # multiprocessing
    with mp.Pool(PROCESSES_COUNT, init_factory) as pool:
        data = pool.map(is_news_worker, joblist)
        pool.close()
        pool.join()

    # prepare result
    news_articles = [fname for fname in data if fname]
    return {"articles": news_articles}


def is_news_worker(filepath):
    """
        Check if html is news article.
        Returns filename if yes, None otherwise
    """
    lang, title = get_lang_and_title(filepath)
    if simpletools.is_news_title(lang, title):
        return os.path.basename(filepath)
    return None


def get_lang_and_title(filepath):
    # read file
    with open(filepath, 'rb') as f:
        page_raw = f.read()

    tree = HTMLParser(page_raw)
    text = ""
    title = tree.css_first("meta[property=\"og:title\"]").attributes['content']

    # detect lang by description
    description = tree.css_first("meta[property=\"og:description\"]")
    if description:
        text = description.attributes['content'].strip()
    # or by title
    if not text:
        text = title

    # detect lang precisiously
    try:
        lang = detect(text)
    except:
        return None, None

    return lang, title

