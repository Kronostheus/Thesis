import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from Utils import Config

URL = "https://manifestoproject.wzb.eu/coding_schemes/mp_v5"

r = requests.get(URL)

r.raise_for_status()

soup = BeautifulSoup(r.content, features="html.parser")

category_list = soup.find("ul", {"class": "list nolist"}).find_all_next("li")

test = category_list[2]

row_list = []

subcategory = ""

for category in category_list:
    code = category.span.get_text(strip=True)
    major = Config.MAN_MAJOR_TOPICS[code[0]]
    minor = category.h3.get_text(strip=True)

    desc = category.p

    if desc:
        description = desc.get_text(strip=True)
    else:
        description = subcategory = minor = minor[3:].split('- comprised of')[0].strip()

    if "." in code:
        minor = subcategory + " >> " + minor

    row_list.append({
        "Major Topic": major,
        "Minor Topic": minor,
        "Code": code,
        "Description": re.sub(r'[^a-zA-Z0-9;,./:()\-\' ]', "", description)
    })

row_list.append({
        "Major Topic": "Header",
        "Minor Topic": "Header",
        "Code": "H",
        "Description": "Header Span"
    })

codebook = pd.DataFrame(row_list, columns=["Major Topic", "Minor Topic", "Code", "Description"], dtype='object')

assert not codebook.isnull().values.any()

codebook.to_csv(Config.SCHEMES_DIR + "MAN_v5.csv", index=False)
