# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../libs")
from selectolax.parser import HTMLParser
from config import FIXED_LANGUAGES
import os
import re
import simpletools


def cmd_languages(src_dir):
    # prepare samples
    joblist = simpletools.collect_files(src_dir)

    # multiprocessing
    data = simpletools.parallel_processor(detect_lang_worker, joblist)

    # prepare result
    pages_by_lang = {}
    for i in data:
        lang, fname = i[0], i[1]
        if lang not in pages_by_lang.keys():
            pages_by_lang[lang] = []
        pages_by_lang[lang].append(fname)

    return pages_by_lang


def detect_lang_worker(filepath):
    # read file
    with open(filepath, 'rb') as f:
        page_raw = f.read()

    tree = HTMLParser(page_raw)
    text = ""

    # detect lang by description
    description = tree.css_first("meta[property=\"og:description\"]")
    if description:
        text = description.attributes['content'].strip()

    # or by title
    if not text:
        title = tree.css_first("meta[property=\"og:title\"]")
        text = title.attributes['content']

    # detect lang precisiously
    try:
        lang = simpletools.detectLang(text, detect_all=True)
    except:
        return None, None

    return lang, os.path.basename(filepath)
