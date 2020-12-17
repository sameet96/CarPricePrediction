import os
import pandas as pd
# cwd = os.('C:\\Users\\samee\\Documents\\carvana\\Data')
ls = os.chdir('C:\\Users\\samee\\Documents\\carvana')
files = os.listdir(ls)
# print(files)

df = pd.read_csv("data.csv")
# print(files)
# df = pd.read_csv('data.csv')

# print(df['price'].describe())

df['price'] = df['price'].str[1:]
print(df['price'].describe())