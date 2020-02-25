import pandas as pd
import numpy as np
import re
import os.path

from nltk.corpus import stopwords, words
from nltk.tokenize import sent_tokenize
from gensim import downloader
from ast import literal_eval
from operator import itemgetter

DATA_DIR = "Data/Coding_Schemes/"

MAN = DATA_DIR + 'MAN_v4.csv'
CAP = DATA_DIR + 'CAP.csv'

cap_df = pd.read_csv(CAP)


def remove_beginning(description):
    """
    The CAP codebook contains a very similar beginning for all descriptions, which I edit out.
    :param description: string containing the description
    :return: Description with its initial part removed
    """
    return re.sub(r'(Includes )?[Ii]ssues (generally )?related (generally )?to', '', description)


def remove_punctuation(description):
    """
    Removes all punctuation, or in a more explicit way, removes anything that.
    :param description: string containing the description
    :return: Description with all punctuation removed
    """
    return re.sub(r'[^\w\s]', '', description)


def remove_stopwords(description):
    """
    Removal of stopwords from the description. Considers the stopwords from the English language.
    :param description: string containing the description
    :return: List with all words from description except those belonging to the stopwords corpus
    """
    return [word for word in description.split() if word not in stopwords.words('english')]


def build_dict(df):
    """
    Builds a dictionary that zips together the Code and respective Description of a topic/category into a dictionary
    :param df: DataFrame of the Coding Scheme
    :return: Dictionary<Code, Description>
    """
    return dict(zip(df.Code, df.Description))


def include_cap_topics(x):
    x.Description += " " + x["Major Topic"] if x["Minor Topic"] in ["General", "Other"] else " " + x["Minor Topic"]
    return x

print("Preprocessing CAP dataset")

cap_df.Description = cap_df.Description.apply(lambda x: remove_beginning(x))
cap_df = cap_df.apply(lambda x: include_cap_topics(x), axis=1)
cap_df.Description = cap_df.Description.apply(lambda x: remove_punctuation(x))
cap_df.Description = cap_df.Description.apply(lambda x: x.strip().lower())
cap_df.Description = cap_df.Description.apply(lambda x: remove_stopwords(x))

# Dictionary with the Code and Description -> Dictionary<Code, Description>
cap_desc_dict = build_dict(cap_df)

man_df = pd.read_csv(MAN)


def remove_includes(description):
    """
    Removes sentences that end with a : since they are mostly useless. (ex. "May include:")
    :param description: string containing the description
    :return: Description without the affected expressions
    """
    return re.sub(r'[A-Z][a-z]+.*:', '', description)


def remove_numbers(description):
    """
    Removes numbers from description which usually correspond to mentioning other categories
    :param description: string containing the description
    :return: Description without numbers
    """
    return re.sub(r'\d+', '', description)


def remove_patterns(description):
    """
    In the Manifesto codebook, the descriptions include some common words and expressions that I would like to remove.
    These include abbreviations and words representing sentiment.
    :param description: string containing the description
    :return: Description with affected expressions removed
    """
    return re.sub(r'\[|\]|e\.g|etc\.|and/or |[Ff]avourable |[Oo]pposition |[Nn]egative |[Pp]ositive |[Aa]ppeal(s)? |\'s'
                  r'[Mm]entions |[Rr]eferences |[Uu]nfavourable |[Ss]upport |[Gg]eneral |[Ll]imiting |Community/|ECs/|'
                  r'[Pp]ositive|[Nn]egative|[Ll]imitation|[Ee]xpansion',
                  '', description)


def remove_empty(description_list):
    """
    Given a list with descriptions, strips edge whitespaces, reduces all characters to lowercase and filters out
    empty strings.
    :param description_list: List containing descriptions
    :return: cleaned list
    """
    desc_list = [description.strip().lower() for description in description_list]
    return list(filter(None, desc_list))


def flatten(description_list, include_repeat=True):
    """
    Flattens list of lists to a single list containing words.
    :param description_list: List containing descriptions
    :param include_repeat: Include repeated words in list (boolean)
    :return: flattened list of words
    """
    flattened = [val for sublist in description_list for val in sublist]
    return flattened if include_repeat else list(set(flattened))


def remove_non_english(word_list):
    """
    Remove words not present in the english dictionary in order to avoid complications with misspells found in the
    descriptions.
    :param word_list: List containing words
    :return: List with non-english words removed
    """
    eng_words = set(words.words())
    return [word for word in word_list if word in eng_words]


print("Preprocessing MAN dataset")

man_df = man_df.apply(lambda x: include_cap_topics(x), axis=1)
man_df.Description = man_df.Description.apply(lambda x: sent_tokenize(x))
man_df.Description = man_df.Description.apply(lambda x: list(map(remove_includes, x)))
man_df.Description = man_df.Description.apply(lambda x: list(map(remove_numbers, x)))
man_df.Description = man_df.Description.apply(lambda x: list(map(remove_patterns, x)))
man_df.Description = man_df.Description.apply(lambda x: list(map(remove_punctuation, x)))
man_df.Description = man_df.Description.apply(lambda x: remove_empty(x))
man_df.Description = man_df.Description.apply(lambda x: list(map(remove_stopwords, x)))
man_df.Description = man_df.Description.apply(lambda x: flatten(x))
man_df.Description = man_df.Description.apply(lambda x: remove_non_english(x))

# Dictionary with the Code and Description -> Dictionary<Code, Description>
man_desc_dict = build_dict(man_df)

# Besides not necessary to recompute CSV unless there are changes, loading the model is very time consuming
if not os.path.exists(DATA_DIR + 'match4.csv'):
    print("Downloading model")

    # Pretrained Word2Vec model
    model = downloader.load('word2vec-google-news-300')

    # Normalize vectors using L2 regularization
    model.init_sims(replace=True)

    sims_dict = {}

    print("Computing distance")

    for cap_code, cap_desc in cap_desc_dict.items():

        # List of Tuples -> Tuple(MAN_CODE, DISTANCE)
        sims = []

        for man_code, man_desc in man_desc_dict.items():

            # Compute Word Mover's Distance between CAP description and all MAN descriptions
            distance = model.wmdistance(cap_desc, man_desc)

            sims.append((man_code, distance))

        # Associate CAP code with computed distances
        sims_dict[cap_code] = sims

    df = pd.DataFrame(sims_dict)
    df.to_csv(DATA_DIR + 'match4.csv', index=False)

matches_df = pd.read_csv(DATA_DIR + 'match4.csv')
corrs = []

for cap_code in matches_df.columns:

    # Read tuples that are in string form inside the CSV
    corr_list = [literal_eval(corr_tuple) for corr_tuple in matches_df[cap_code].to_numpy()]

    # Sort the distances from lowest to highest and get the best five
    top_5 = sorted(corr_list, key=itemgetter(1))[:5]

    row_lst = [cap_code]

    for man_code, corr in top_5:

        # For readability sake, include the long-form name of MAN category
        man_name = man_df[man_df.Code == man_code]["Minor Topic"].values[0]

        # Each entry has the CODE, NAME and DISTANCE associated with the Manifesto category
        row_lst.extend([man_code, man_name, corr])

    corrs.append(row_lst)

corr_df = pd.DataFrame(corrs, columns=['CAP', '1 CODE', '1 NAME', '1 CORR', '2 CODE', '2 NAME', '2 CORR',
                                       '3 CODE', '3 NAME', '3 CORR', '4 CODE', '4 NAME', '4 CORR',
                                       '5 CODE', '5 NAME', '5 CORR'])

corr_df.to_csv(DATA_DIR + 'best_matches4.csv', index=False)
