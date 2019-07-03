import pandas as pd
import statistics
df1 = pd.read_csv("/root/alpha/git/mine/data_science/data/01JAN/SBIN.csv",header=None)
df = df1[6]
statistics.mean(df)
df #meannn