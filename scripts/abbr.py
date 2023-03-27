#! /usr/bin/env python3
"""
abbr.py - substitute journal names in .bib file with ADS journal abbreviations.
Last Modified: 2021.04.14

Copyright(C) 2020 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details

Usage:
Modify the value of 'bib' to the absolute path of your .bib file, then execute this script.
"""
import os
import re

import requests
from bs4 import BeautifulSoup

# Replace the following strings with
# bib - absolute path to your .bib file
# cache - absolute path to 'adsbibcode.txt'
bib = "/Users/xshaokun/Documents/latex/references.bib"
cache = "/Users/xshaokun/Documents/latex/adsbibcode.txt"

# Check if cache of adsbibcode exists. If not, cache it from http://adsabs.harvard.edu/abs_doc/journals.html.
if os.path.isfile(cache):
    abbr = {}
    with open(cache) as f:
        for line in f.readlines():
            line = line.strip()
            k = line.split(":")[0]
            v = line.split(":")[1]
            abbr[k] = v
else:
    url = "http://adsabs.harvard.edu/abs_doc/journals.html"
    response = requests.get(url)
    text = response.content
    soup = BeautifulSoup(text, "html.parser")
    journals = []
    for string in soup.pre.stripped_strings:
        journals.append(string)
    abb = []
    name = []
    for i in range(0, len(journals), 2):
        abb.append(journals[i])
        name.append(journals[i + 1])
    abbr = dict(zip(name, abb, strict=True))
    with open(cache, "w") as f:
        for k, v in abbr.items():
            f.write(k + ":" + v + "\n")


# Import journal abbreviation macros defined in AASTeX v6.3.1, url: https://journals.aas.org/aastexguide/#abbreviations
# url = 'https://journals.aas.org/aastexguide/#abbreviations'
# header = {
#   "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
#   "X-Requested-With": "XMLHttpRequest"
# }
#
# response = requests.get(url, headers=header)
# dfs = pd.read_html(response.text)[1]
# Journal names in this table are differenet from those in ADS (without 'the').
# Currently don't include it in abbreviation dictionary.


# Append some abbreviations which are not included
others = {"The Astrophysical Journal Letters": r"\apjl"}

for k, v in others.items():
    abbr[k] = v


# Replace the journal iterm  with abbreviation according to ADS Journal Abbreviations.
# The only  exception is arXiv whose abbreviation in adsbibcode.txt is wrong
with open(bib) as f1, open("temp.bib", "w") as f2:
    for line in f1.readlines():
        if "journal" in line:
            obj = re.search(r"{(.*)}", line, re.I)
            key = obj.group(1)
            if abbr.get(key):
                line = line.replace(key, abbr.get(key))
            line = re.sub(r"{(arxiv.*)}", "{arXiv e-prints}", line, flags=re.I)
            line = re.sub("&", r"\&", line)
        f2.write(line)
    os.remove(bib)
    os.rename("temp.bib", bib)
