import requests
import re
import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "Data/Coding_Schemes/"
URL = "https://manifestoproject.wzb.eu/coding_schemes/mp_v5"

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

test = category_list[2]

row_list = []

subcategory = ""

for category in category_list:
    code = category.span.get_text(strip=True)
    major = MAJOR_TOPICS[code[0]]
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

codebook.to_csv(DATA_DIR + "MAN.csv", index=False)
