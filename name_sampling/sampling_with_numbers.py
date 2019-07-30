
# coding: utf-8

import pandas as pd
import numpy as np
import os

fileDir = os.path.abspath(os.path.dirname(__file__))
korean_csv = os.path.join(fileDir, "name_kor_combined_space.csv")
usa_csv = os.path.join(fileDir, "name_usa_combined_space.csv")

korean_df = pd.read_csv(korean_csv)
korean_df = korean_df[['name']]
usa_df = pd.read_csv(usa_csv)
usa_df = usa_df[['name']]
print("length of Korean dataframe: ", len(korean_df), "\nLength of English dataframe: ",len(usa_df))

kor_num = int(input("How many Korean names do you want?: "))
eng_num = int(input("How many English names do you want?: "))
usa_df = usa_df.sample(n=eng_num).sort_values(by=['name']).reset_index(drop=True)
korean_df = korean_df.sample(n=kor_num).sort_values(by=['name']).reset_index(drop=True)

usa_df['n_or_f'] = 'f'
usa_df['name_len'] = usa_df['name'].str.len()
usa_df.head()

korean_df['n_or_f'] = 'n'
korean_df['name_len'] = korean_df['name'].str.len()
korean_df.head()


combined_df = pd.concat([usa_df, korean_df])
combined_df = combined_df.reset_index(drop=True)
print(combined_df)

combined_df.to_csv("origin_in.csv")

