import glob
import pandas as pd
from abc import ABC, abstractmethod


class DataProcessor(ABC):
    def __init__(self, path):
        self.path = path
        self.dataframes = self.get_dataframes()

    def get_dataframes(self):
        """
        Dictionary containing dataframes and path to CSV that originate them.
        :return: Dictionary<path, dataframe>
        """
        return {file: pd.read_csv(file) for file in glob.glob(self.path + '*.csv')}

    @staticmethod
    @abstractmethod
    def drop(df):
        """
        Drop columns that are irrelevant for our purposes
        :param df: Dataframe
        :return: None
        """
        pass

    @staticmethod
    @abstractmethod
    def rename(df):
        """
        Rename dataframe columns into [Text, Code] and placing them in the correct order if needed.
        :param df: Dataframe
        :return: None
        """
        pass

    @staticmethod
    @abstractmethod
    def convert(df):
        """
        Child-class specific
        :param df: Dataframe
        :return: None
        """
        pass

    @staticmethod
    def reduce(man_code):
        """
        In MAN codes where there is sentiment, we only keep one. Code now acts as a general statement from both codes.
        Ex: Education Limitation (507) vs Education Expansion (506) => Education (506)
        :param man_code: MAN code
        :return: MAN code
        """
        reduce_dict = {'102': '101', '105': '104', '109': '107', '110': '108', '204': '203', '407': '406', '505': '504',
                       '507': '506', '602': '601', '604': '603', '608': '607', '702': '701'}

        return reduce_dict[man_code] if man_code in reduce_dict.keys() else man_code

    @abstractmethod
    def execute(self):
        """
        Run CSV preprocessing and save.
        :return: None
        """
        pass


class CAPProcessor(DataProcessor):
    def __init__(self, path):
        super(CAPProcessor, self).__init__(path)

    @staticmethod
    def drop(df):
        df.drop(['id', 'year', 'month', 'day', 'majortopic'], inplace=True, axis=1)

    @staticmethod
    def rename(df):
        c = df.columns
        # Swap columns [Code, Text] -> [Text, Code]
        df[[c[0], c[1]]] = df[[c[1], c[0]]]
        df.columns = ['Text', 'Code']

    @staticmethod
    def convert(df):
        """
        Convert CAP codes into MAN codes according to previously found correspondents.
        :param df: Dataframe
        :return: None
        """
        corr_df = pd.read_csv('Data/Coding_Schemes/cap_to_man.csv', dtype='object')
        corrs = dict(zip(corr_df.Code, corr_df.MAN))
        # Some codes have no source, thus, I consider them a mistake and drop the row.
        df.Code = df.Code.apply(lambda x: corrs[str(x)] if str(x) in corrs.keys() else pd.NaT)

    def execute(self):
        for path, df in self.dataframes.items():
            self.drop(df)
            self.rename(df)
            self.convert(df)
            df.dropna(inplace=True)
            df.Code = df.Code.apply(lambda x: self.reduce(x))
            df.Text = df.Text.apply(lambda x: str(x).strip())
            df.to_csv(path, index=False)


class MANProcessor(DataProcessor):
    def __init__(self, path):
        super(MANProcessor, self).__init__(path)

    @staticmethod
    def drop(df):
        df.drop(['eu_code'], inplace=True, axis=True)

    @staticmethod
    def rename(df):
        df.columns = ['Text', 'Code']

    @staticmethod
    def convert(df):
        """
        Convert from MAN v5 to v4 where necessary
        :param df: Dataframe
        :return: None
        """
        df.Code = df.Code.apply(lambda x: str(x).split('.')[0])

    def execute(self):
        for path, df in self.dataframes.items():
            self.drop(df)
            self.rename(df)
            df.Code.fillna('H', inplace=True)
            self.convert(df)
            df.Code = df.Code.apply(lambda x: self.reduce(x))
            df.Text = df.Text.apply(lambda x: str(x).strip())
            df.to_csv(path, index=False)


processes = [CAPProcessor('Data/Spain_Media/'),
             MANProcessor('Data/Portugal_Manifestos/'),
             MANProcessor('Data/Brazil_Manifestos/')]

for proc in processes:
    proc.execute()