import requests
import re
import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "Data/Coding_Schemes/"
URL = "https://manifestoproject.wzb.eu/coding_schemes/mp_v4"

MAJOR_TOPICS = {
    "0": "General",
    "1": "External Relations",
    "2": "Freedom and Democracy",
    "3": "Political System",
    "4": "Economy",
    "5": "Welfare and Quality of Life",
    "6": "Fabric of Society",
    "7": "Social Groups",
    "H": "Header"
}

r = requests.get(URL)

r.raise_for_status()

soup = BeautifulSoup(r.content, features="html.parser")

category_list = soup.find("ul", {"class": "list nolist"}).find_all_next("li")

row_list = []

subcategory = ""

for category in category_list:
    code = category.span.get_text(strip=True)
    major = MAJOR_TOPICS[code[0]]
    minor = category.h3.get_text(strip=True)
    description = category.p.get_text().split('\n')

    desc_list = [d if d[0].isalnum() else d[1:].strip() for d in description if d]
    desc_cln = [re.sub(r' +', ' ', d) for d in desc_list]

    row_list.append({
        "Major Topic": major,
        "Minor Topic": minor,
        "Code": code,
        "Description": desc_cln
    })

row_list.append({
        "Major Topic": "Header",
        "Minor Topic": "Header",
        "Code": "H",
        "Description": "Header Span"
    })

codebook = pd.DataFrame(row_list, columns=["Major Topic", "Minor Topic", "Code", "Description"], dtype='object')

assert not codebook.isnull().values.any()

codebook.to_csv(DATA_DIR + "MAN_v4.csv", index=False)
