import requests
import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "Data/Coding_Schemes/"
URL = "https://www.comparativeagendas.net/pages/master-codebook"

r = requests.get(URL)

r.raise_for_status()

soup = BeautifulSoup(r.content, features="html.parser")

category_div = soup.find('ul')

domains = [child for child in category_div.children if child != '\n']

text_last = domains[-1].get_text().split("\n")
build_tag = "<li><strong>" + domains[-2].get_text() + "</strong><ul><li>" + text_last[1] + "<em>" + text_last[3] + "</em></li></ul></ul>"
fixed_domains = domains[:-2]
fixed_domains.append(BeautifulSoup(build_tag, features="html.parser"))

row_list = []

for domain in fixed_domains:
    major_topic = domain.find('strong').get_text().strip().split()[1:]
    for minor in list(domain.find_all('li')):
        description = minor.get_text(strip=True).split(':')[-1]
        minor_topic = minor.get_text(strip=True).split('Desc')[0].split(':')

        if len(minor_topic[0]) > 4:
            continue

        row_list.append({
            "Major Topic": ' '.join(major_topic),
            "Minor Topic": minor_topic[1].strip(),
            "Code": minor_topic[0],
            "Description": description.strip()
        })

codebook = pd.DataFrame(row_list, columns=["Major Topic", "Minor Topic", "Code", "Description"], dtype='object')
codebook["Minor Topic"] = codebook["Minor Topic"].apply(lambda x: x.replace('R&D', 'Research and Development'))
codebook["Minor Topic"] = codebook["Minor Topic"].apply(lambda x: x.replace('&', 'and'))

assert codebook.shape[0] == 213 and codebook.shape[1] == 4
assert not codebook.isnull().values.any()

codebook.to_csv(DATA_DIR + "CAP.csv", index=False)
