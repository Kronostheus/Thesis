import glob
import pandas as pd
import numpy as np
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
                       '507': '506', '602': '601', '604': '603', '608': '607', '702': '701', 'H': '999'}

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

    @staticmethod
    def fix_dataframe(df):
        offset = 0
        for index, row in df.iterrows():
            text = str(row.title)
            if '\t' in text:
                index += offset
                split_text = text.split('\r\n')
                df.loc[index, 'title'] = split_text[0]
                new_df = pd.DataFrame([tmp_row.split('\t') for tmp_row in split_text[1:]], columns=df.columns)
                offset += new_df.shape[0]
                df.iloc[:] = pd.concat([df.iloc[:index+1], new_df, df.iloc[index+1:]]).reset_index(drop=True)

    @staticmethod
    def fix_breaks(text):
        return " ".join(element.strip() for element in text.split('\r\n')) if '\r\n' in str(text) else str(text)

    def execute(self):
        for path, df in self.dataframes.items():
            self.fix_dataframe(df)
            self.drop(df)
            self.rename(df)
            self.convert(df)
            df.dropna(inplace=True)
            df.Code = df.Code.apply(lambda x: self.reduce(x))
            df.Text = df.Text.apply(lambda x: str(x).strip())
            df.Text.apply(lambda x: self.fix_breaks(x))
            df.Text.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)
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

    @staticmethod
    def clean(row):
        code = row.Code
        text = row.Text
        if len(str(text).split()) > 150 and str(code) in ('nan', 'H'):
            row.Code = np.nan
        if str(code) == 'nan' and len(str(text).split()) < 20:
            row.Code = 'H'
        return row

    def execute(self):
        for path, df in self.dataframes.items():
            self.drop(df)
            self.rename(df)
            df = df.apply(lambda x: self.clean(x), axis=1)
            df.dropna(inplace=True)
            self.convert(df)
            df.Code = df.Code.apply(lambda x: self.reduce(x))
            df.Text = df.Text.apply(lambda x: str(x).strip())
            df.Text.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)
            df.to_csv(path, index=False)


processes = [MANProcessor('Data/Italy_Manifestos/')]

for proc in processes:
    proc.execute()
