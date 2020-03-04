import pandas as pd

DATA_DIR = 'Data/Coding_Schemes/'

def fill_individual(x):
    """
    Based on NA values, build dictionary of CAP codes whose MAN correspondence should be individually hand-picked
    :param x: row of dataframe
    :return: updated row
    """
    individuals = {
        '105': '414',                   # National Budget               -> Economic Orthodoxy
        '599': '701',                   # Labor - Other                 -> Labour Groups
        '900': '607',                   # Immigration                   -> Multiculturalism
        '1000': '411',                  # Transportation - General      -> Technology and Infrastructure
        '1523': '504',                  # Disaster Relief               -> Welfare State
        '1701': '411',                  # Space                         -> Technology and Infrastructure
        '2010': '304',                  # Scandals                      -> Political Corruption
        '2102': '607'                   # Indigenous Affairs            -> Multiculturalism
    }

    # I also ensure that all codes have the same format (XXX)
    x.MAN = individuals[x.Code] if x.Code in individuals.keys() else str(x.MAN).split('.')[0]
    return x


def fill_groups(x):
    """
    Some CAP domains can be grouped up as the same. If an exception should be made, individual codes are processed
    beforehand, allowing greater control.
    :param x: row of dataframe
    :return: updated row
    """
    groups = {'80': '411',              # Energy        (6)              -> Technology and Infrastructure
              '14': '411',              # Housing       (7)              -> Technology and Infrastructure
              '16': '104',              # Defense       (2)              -> Military
              '21': '501'}              # Public Lands  (2)              -> Environmental Protection

    if x.MAN == 'nan':
        # Treat the first 2 digits of the CAP code as a group. THIS CAN HAVE PROBLEMS FOR OTHER CODES! (ex: 199 vs 1901)
        cap_group = x.Code[:2]
        x.MAN = groups[cap_group] if cap_group in groups.keys() else x.MAN
    return x


def spanish_media_codes():
    """
    Spanish Media files seem to have been coded with its own codebook, heavily based on CAP master codebook. For the
    most part, the correspondences between CAP master codebook and MAN codebook apply. Some CAP codes are not included
    in Spanish Media codebook. Eight new codes were added to the existing CAP topics, three from an existing CAP Major
    Topic (Culture) and five created within new Major Topics (Climate (1), Sports (1) and Death Notices (3)).

    This method creates the correspondence between the eight new topics and their respective MAN code. This is done in
    the shape of a DataFrame that is later appended to the original correspondence DataFrame.
    :return: DataFrame containing correspondence between unique spanish media codes and MAN codes
    """
    corrs = {
        '2301': '502',                  # Culture: Cinema, Theatre, etc. -> Culture
        '2302': '502',                  # Culture: Books                 -> Culture
        '2399': '502',                  # Culture: Others                -> Culture
        '2700': '501',                  # Climate: General               -> Environmental Protection
        '2900': '502',                  # Sports: General                -> Culture
        '3001': '000',                  # Death Notices: Natural Death   -> General
        '3002': '000',                  # Death Notices: Violent Death   -> General
        '3099': '000'                   # Death Notices: Others          -> General
    }

    return pd.DataFrame(corrs.items(), columns=['Code', 'MAN'])


def reduce(man_code):
    """
    In MAN codes where there is sentiment, we only keep one. Code now acts as a general statement from both codes.
    Ex: Education Limitation (507) vs Education Expansion (506) => Education (506)
    :param man_code: MAN code
    :return: MAN code
    """
    reduce_dict = {
        '102': '101',
        '105': '104',
        '109': '107',
        '110': '108',
        '204': '203',
        '407': '406',
        '505': '504',
        '507': '506',
        '602': '601',
        '604': '603',
        '608': '607',
        '702': '701'
    }

    return reduce_dict[man_code] if man_code in reduce_dict.keys() else man_code


df = pd.read_csv(DATA_DIR + 'cap_to_man.csv', dtype='object')

df = df.apply(lambda x: fill_individual(x), axis=1)         # Individuals first
df = df.apply(lambda x: fill_groups(x), axis=1)             # Groups second
df = df.append(spanish_media_codes(), ignore_index=True)    # Spanish Media additional codes
df.MAN = df.MAN.apply(lambda x: reduce(x))

df.to_csv(DATA_DIR + 'cap_to_man.csv', index=False)

cap = pd.read_csv(DATA_DIR + 'CAP.csv', dtype='object')
man = pd.read_csv(DATA_DIR + 'MAN_v4.csv', dtype='object')


def verbose(row):
    """
    For visual confirmation of correspondence, I also include a CSV file with each code named explicitly.
    :param row: row of dataframe
    :return: new row
    """
    spanish_codes = {
        '2301': 'Culture: Cinema, theatre, music and dance',
        '2302': 'Culture: Publication of books and literary works',
        '2399': 'Culture: Others',
        '2700': 'Climate: General',
        '2900': 'Sports: General',
        '3001': 'Death Notices: Natural Death',
        '3002': 'Death Notices: Violent Death',
        '3099': 'Death Notices: Others'
    }

    row.Code = spanish_codes[row.Code] if row.Code in spanish_codes.keys() \
        else ': '.join(cap[cap.Code == row.Code][["Major Topic", "Minor Topic"]].iloc[0].tolist())

    # MAN codes are smaller in range and more explicit
    row.MAN = man[man.Code == row.MAN]["Minor Topic"].tolist()[0]

    return row


verbose_df = df.apply(lambda x: verbose(x), axis=1)
verbose_df.to_csv(DATA_DIR + 'cap_to_man_verbose.csv', index=False)
