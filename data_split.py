import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from functools import reduce

DATA_DIR = 'Data/'
CLASS_DIR = DATA_DIR + 'Classification/'

CLASS_PATH = Path(CLASS_DIR)
CLASS_PATH.mkdir(exist_ok=True)

TRAIN_PATH = Path(str(CLASS_PATH) + '/Train/')
TRAIN_PATH.mkdir(exist_ok=True)

TEST_PATH = Path(str(CLASS_PATH) + '/Test/')
TEST_PATH.mkdir(exist_ok=True)

# VAL_PATH = Path(str(CLASS_PATH) + '/Val/')
# VAL_PATH.mkdir(exist_ok=True)

SPAIN_DATA = Path(DATA_DIR + '/Spain_Media/').glob('*.csv')
PORTUGAL_DATA = Path(DATA_DIR + '/Portugal_Manifestos/').glob('*.csv')
BRAZIL_DATA = Path(DATA_DIR + '/Spain_Media/').glob('*.csv')

data = pd.DataFrame()
for data_generator in [PORTUGAL_DATA, SPAIN_DATA, BRAZIL_DATA]:
    data = pd.concat([data, pd.concat([pd.read_csv(path) for path in data_generator])])


def bundle_count(df, prefix):
    """
    Build dataframe containing the Codes, their respective occurrence Count and their Percentage within the dataframe.
    :param df: Dataframe to process
    :param prefix: Prefix to append to each columns
    :return: Dataframe with columns: Code, Count(Code), Percentage(Code)
    """
    joined = pd.merge(df.Code.value_counts().reset_index(),
                      df.Code.value_counts(normalize=True).apply(lambda x: x * 100).reset_index(),
                      on='index')
    joined.columns = ['Code', prefix + '_Count', prefix + '_Perc']
    return joined


def check_stratify(main, dfs):
    """
    Performs a series of checks to the splitting of data from one main dataframe and a list of dataframes. Also creates
    a CSV file containing information about the split. Ensures that the data was correctly split in a stratified manner.
    :param main: Main dataframe from whence the splitting originated
    :param dfs: Dataframes that result of the splitting of the main dataframe
    :return:
    """

    df_names = {0: 'Train', 1: 'Test'}

    df_lst = [bundle_count(main, "Main")]

    for i in range(len(dfs)):
        df = dfs[i]

        # Throw exception if percentages of codes are not similar to a given threshold
        assert np.allclose(main.Code.value_counts(normalize=True),
                           df.Code.value_counts(normalize=True),
                           rtol=1e-3, atol=1e-5)

        df_lst.append(bundle_count(df, df_names[i]))

    # Inner Join all dataframes in df_lst
    results = reduce(lambda df1, df2: pd.merge(df1, df2, on='Code'), df_lst)

    # Throw exception if split dataframes would not rebuild into the original one. Also check if percentages add to 100%
    assert np.allclose(results.sum()[1:], [main.shape[0], 100,
                                           int(main.shape[0] * 0.6), 100,
                                           main.shape[0] - int(main.shape[0] * 0.6), 100])

    results.Code = results.Code.astype('object')
    results.to_csv(CLASS_DIR + 'dataset_split.csv', index=False)


# Split dataframe at the 60% mark and 80% mark resulting in a (60, 20, 20) split
train, test = train_test_split(data, train_size=0.6, random_state=1, shuffle=True, stratify=data.Code)

# Validate results
check_stratify(data, [train, test])

train.to_csv(str(TRAIN_PATH) + '/train.csv', index=False)
# val.to_csv(str(VAL_PATH) + '/val.csv', index=False)
test.to_csv(str(TEST_PATH) + '/test.csv', index=False)
