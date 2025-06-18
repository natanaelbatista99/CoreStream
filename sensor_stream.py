import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,LabelEncoder, MinMaxScaler, OneHotEncoder)
from sklearn.preprocessing import Normalizer, MaxAbsScaler , RobustScaler, PowerTransformer

# read kddcup_4m
sensor = pd.read_csv('datasets/sensor_2m.csv')
print(sensor.head(10))
print(sensor.shape)

sensor.drop('class', axis=1, inplace=True)
sensor.drop_duplicates(subset=None, keep="first", inplace=True)
print(sensor.head(10))
print(sensor.shape)

sensor.to_csv('datasets/sensor_stream.csv', sep=',', index = False)