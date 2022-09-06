from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pandas as pd
import glob
import os
import numpy as np

# Apagar se o arquivo existe.
if os.path.exists("tweets.csv"):
    os.remove("tweets.csv")

# Cria um csv novo.
f = open("tweets.csv", "x")

# Abre a pasta tweets, lê todos os arquivos com a extensão txt e faz um csv.
with open('tweets.csv', 'a') as csv_file:
    for path in glob.glob('./tweets/*.txt'):
        with open(path) as txt_file:
            txt = txt_file.read() + '\n'
            csv_file.write(txt)

# Criação do dataframe.
df = pd.read_csv('tweets.csv', encoding='cp1252', on_bad_lines='skip')
df = df.drop_duplicates()


# Realização do IDF
cv = CountVectorizer()
word_count_vector=cv.fit_transform(df)


tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)

df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
df_idf.sort_values(by=['idf_weights'])
print(df_idf)

# Realização do TF-IDF
count_vector=cv.transform(df) 
tf_idf_vector=tfidf_transformer.transform(count_vector)
feature_names = cv.get_feature_names()

first_document_vector=tf_idf_vector[0]
df_tfidf = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
df_tfidf.sort_values(by=["tfidf"],ascending=False)
print(df_tfidf)