import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,LabelEncoder, MinMaxScaler, OneHotEncoder)
from sklearn.preprocessing import Normalizer, MaxAbsScaler , RobustScaler, PowerTransformer

feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","attack"]

# read kddcup_4m
kddcup_data = pd.read_csv('datasets/kddcup_4m', names=feature)
print(kddcup_data.head(10))
print(kddcup_data.shape)

# LABELS attack
labels = set(kddcup_data['attack'].values)
print(labels)
print(len(labels))

# ENCODER
# attack
kddcup_data['attack_state'] = LabelEncoder().fit_transform(kddcup_data["attack"])
# protocol_type
kddcup_data['protocol_type'] = LabelEncoder().fit_transform(kddcup_data["protocol_type"])
kddcup_data['service']       = LabelEncoder().fit_transform(kddcup_data["service"])
kddcup_data['flag']          = LabelEncoder().fit_transform(kddcup_data["flag"])
#print(kddcup_data.flag.value_counts())

# Drop duplicates
kddcup_data.drop_duplicates(subset=None, keep="first", inplace=True)
print(kddcup_data.shape)
print(kddcup_data.attack.value_counts())

# Save the labels
labels = kddcup_data[['attack', 'attack_state']]
print(labels.head())
labels.to_csv('datasets/kddcup_labels.csv', sep=",", index = False)
sys.exit(1)

# Drop attack and attack_state
df = kddcup_data.drop(['attack', 'attack_state'], axis = 1)
print(df.info())
print(df.T[0])

df.to_csv('datasets/kddcup_1m.csv', sep=',', index = False)