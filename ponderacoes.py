import pandas as pd
import glob
import os
import numpy as np
from collections import Counter
from collections import defaultdict

if os.path.exists("tweets.csv"):
    os.remove("tweets.csv")

f = open("tweets.csv", "x")

with open('tweets.csv', 'a') as csv_file:
    for path in glob.glob('./tweets/*.txt'):
        with open(path) as txt_file:
            txt = txt_file.read() + '\n'
            csv_file.write(txt)

words_set = set()
df = pd.read_csv('tweets.csv', encoding='cp1252')

corpus = ['data science is one of the most important fields of science',
          'this is one of the best data science courses',
          'data scientists analyze data' ]

for doc in corpus:
    words = doc.split(' ')
    words_set = words_set.union(set(words))

print("Dados:")
print('Número de palavras:',len(words_set))
print('Palavras: \n', words_set)
print("\n")

n_docs = len(corpus)    
n_words_set = len(words_set) 

df_tf = pd.DataFrame(np.zeros((n_docs, n_words_set)), columns=words_set)

# TF:
print("Resultado do TF")
for i in range(n_docs):
    words = corpus[i].split(' ')
    for w in words:
        df_tf[w][i] = df_tf[w][i] + (1 / len(words))
        
print(df_tf)
f = open("tf.txt", "w")
f.write(df_tf.to_string())
print("\n")

# IDF:
print("Resultado do IDF")
idf = {}

for w in words_set:
    k = 0
    
    for i in range(n_docs):
        if w in corpus[i].split():
            k += 1
            
    idf[w] =  np.log10(n_docs / k)
    
    print(f'{w:>15}: {idf[w]:>10}')

print("\n")

# TF-IDF
print("Resultado do TF-IDF")
df_tf_idf = df_tf.copy()

for w in words_set:
    for i in range(n_docs):
        df_tf_idf[w][i] = df_tf[w][i] * idf[w]
        
print(df_tf_idf)
f = open("df_tf_idf.txt", "w")
f.write(df_tf_idf.to_string())