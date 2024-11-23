import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso



class NameTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        top_brands = X.name.map(lambda x: x.split()[0]).value_counts()
        self.other_brands = top_brands[top_brands < 10].index
        return self

    def transform(self, X):
        X.loc[:, 'brand'] = X.name.apply(
            self.calc_brand, other_brands=self.other_brands
        )
        return X.drop('name', axis=1)
    
    def calc_brand(self, value, other_brands):
        brand = value.split()[0]
        return brand if brand not in self.other_brands else 'Other'
    

class ChoiceFeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        lasso = Lasso(alpha=10)
        lasso.fit(X, y)
        self.grid_future = lasso.coef_ != 0
        return self

    def transform(self, X):
        return X[:, self.grid_future]


def calc_engine_columns(df):
    def calc_num_column(value):
        if value is np.nan:
            return np.nan
        value = value.split()[0]
        try:
            value = float(value)
        except:
            print(f"Значение - {value} заменено на NaN")
            value = np.nan 
        finally:
            return value
    df = df.map(calc_num_column)
    df['max_power_to_engine'] = df['max_power'] / df['engine']
    return df


def calc_torque(df):
    def torque_rpm(value):
        if value is np.nan:
            return (np.nan, np.nan)
        value = value.replace(',', '')
        nums = re.findall(r'\d+(?:\.\d+)?', value)
        value_nm = nums[0]
        value_rpm = nums[-1]
        nm = re.search(r'(Nm)|(kgm)', value, flags=re.IGNORECASE)

        if nm and nm[0] == 'kgm':
            value_nm = float(value_nm) * 9.81
        
        return value_nm, value_rpm
    
    df = df.map(torque_rpm)
    df = df.assign(
        torque=df['torque'].apply(lambda x: x[0]).astype(float),
        max_torque_rpm=df['torque'].apply(lambda x: x[1]).astype(float),
    )
    df['torque_log'] = np.log(df['torque'])
    return df
    

def calc_km_driven(df):
    df['km_driven_inverse'] = 1 / df['km_driven']
    df['km_driven_log'] = np.log(df['km_driven'])
    return df
