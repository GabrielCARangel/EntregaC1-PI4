from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pandas as pd
import glob
import os
import numpy as np

if os.path.exists("tweets.csv"):
    os.remove("tweets.csv")

f = open("tweets.csv", "x")

with open('tweets.csv', 'a') as csv_file:
    for path in glob.glob('./tweets/*.txt'):
        with open(path) as txt_file:
            txt = txt_file.read() + '\n'
            csv_file.write(txt)

df = pd.read_csv('tweets.csv', encoding='cp1252', on_bad_lines='skip')
df = df.drop_duplicates()

print(df)

cv = CountVectorizer()
word_count_vector=cv.fit_transform(df)
df_array = df.toarray()

print(word_count_vector)

tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)
tfidf_transformer

# print idf values 
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
# sort ascending 
df_idf.sort_values(by=['idf_weights'])

print(df_idf)