import pandas as pd

DATA_DIR = 'Data/Coding_Schemes/'


def fill_individual(x):
    """
    Based on NA values, build dictionary of CAP codes whose MAN correspondence should be individually hand-picked
    :param x: row of dataframe
    :return: updated row
    """
    individuals = {
        '105': '414',
        '599': '701',
        '900': '607',
        '1000': '411',
        '1507': '605',
        '1523': '504',
        '1701': '411',
        '1927': '107',
        '2010': '304',
        '2102': '607'
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
    groups = {'80': '411',
              '14': '411',
              '16': '104',
              '21': '501'}

    if x.MAN == 'nan':
        # Treat the first 2 digits of the CAP code as a group. THIS CAN HAVE PROBLEMS FOR OTHER CODES! (ex: 199 vs 1901)
        cap_group = x.Code[:2]
        x.MAN = groups[cap_group] if cap_group in groups.keys() else x.MAN
    return x


def spanish_media_codes():
    corrs = {
        '2301': '502',
        '2302': '502',
        '2399': '502',
        '2700': '501',
        '2900': '502',
        '3001': '000',
        '3002': '000',
        '3099': '000'
    }

    return pd.DataFrame(corrs.items(), columns=['Code', 'MAN'])

df = pd.read_csv(DATA_DIR + 'cap_to_man.csv', dtype='object')

df = df.apply(lambda x: fill_individual(x), axis=1)         # Individuals first
df = df.apply(lambda x: fill_groups(x), axis=1)             # Groups second
df = df.append(spanish_media_codes(), ignore_index=True)    # Spanish Media additional codes

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
