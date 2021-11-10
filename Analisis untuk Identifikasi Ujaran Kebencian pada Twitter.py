#!/usr/bin/env python
# coding: utf-8

# In[277]:


pip install missingno


# In[83]:


pip install wordcloud


# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import ast


# In[100]:


import seaborn as sns


# In[128]:


import unicodedata
from nltk.corpus import stopwords# add appropriate words that will be ignored in the analysis
ADDITIONAL_STOPWORDS = ['covfefe']
import matplotlib.pyplot as plt


# In[59]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[278]:


import matplotlib.pyplot as plt
import missingno as msno


# In[ ]:


from sklearn.svm import SVC
from NLP_Models import modelling as mdg


# In[4]:


from NLP_Models import TextMining as tm


# In[5]:


from NLP_Models import CleanText as ct


# In[6]:


from NLP_Models import modelling as mdg


# In[172]:


from NLP_Models import model_prediction as mo


# # 1. Pelabelan Data HateSpeech dan Data nonHateSpeech

# DATA HateSpeech

# In[195]:


dh = pd.read_json('/home/fani/Documents/NLP/NLP_Models/data/tugas/hateiniya.json', lines = True)


# In[197]:


dh.keys()


# In[135]:


dh = dh[['created_at', 'date', 'time',              'user_id', 'username', 'name',                 'tweet','mentions', 'urls', 'photos', 'replies_count',                      'retweets_count', 'likes_count', 'hashtags',                         'link', 'retweet', 'quote_url',                             'near', 'geo', 'source', 'user_rt_id', 'user_rt',                                 'retweet_id', 'reply_to', 'retweet_date']]


# In[136]:


dh['Label'] = 'hatespeech'


# In[137]:


dh


# DATA nonHateSpeech

# In[138]:


dnh = pd.read_json('/home/fani/Documents/NLP/NLP_Models/data/tugas/bukanhate.json', lines = True)


# In[139]:


dnh.keys()


# In[140]:


dnh = dnh[['created_at', 'date', 'time',              'user_id', 'username', 'name',                 'tweet','mentions', 'urls', 'photos', 'replies_count',                      'retweets_count', 'likes_count', 'hashtags',                         'link', 'retweet', 'quote_url',                             'near', 'geo', 'source', 'user_rt_id', 'user_rt',                                 'retweet_id', 'reply_to', 'retweet_date']]


# In[141]:


dnh['Label'] = 'nonhatespeech'


# In[142]:


dnh


# Menggabungkan data hateSpeech dengan nonHateSpeech

# In[192]:


data = pd.concat([dnh, dh])


# In[145]:


data = data.reset_index(drop= True) 


# In[207]:


data.rename(columns={'tweet':'text'}, inplace=True)


# Cleaning Data

# In[ ]:


data = ct.cleanningtext(data = data, both = True, onlyclean = False, sentiment = False)


# In[ ]:


data.to_json('./NLP_Models/data/Kel4dataFinalCleanttext.json', orient='records')


# # 2. Menghitung banyaknya masing-masing label kategori dalam dataset yang ditampilkan dalam bentuk dataframe

# In[258]:


data = pd.read_json('/home/fani/Documents/NLP/NLP_Models/data/Kel4dataFinalCleanttext.json')


# In[259]:


data1 = data[['text', 'cleaned_text','Label']]


# In[261]:


data[:10]


# In[151]:


data1.groupby('Label').count().transpose()


# # 3. Analisis Data Eksploratif

# In[16]:


data.info()


# In[17]:


np.sum(data.isnull().any(axis=1)) #noise


# In[274]:


data.describe()


# Retweet terbanyak

# In[265]:


df = data[['retweets_count','text']]


# In[266]:


df.loc[df['retweets_count'].idxmax()]


# Like terbanyak

# In[262]:


like = data[['likes_count','text']]


# In[264]:


like.loc[like['likes_count'].idxmax()]


# Reply terbanyak

# In[271]:


reply = data[['replies_count','text']]


# In[272]:


reply.loc[reply['replies_count'].idxmax()]


# Data Label

# In[273]:


plt.figure(figsize=(8,6))
p = sns.countplot(x="Label", data=data1)


# In[279]:


data = data.loc[:,['username','user_id']]
data.sort_values(by='user_id',ascending=False,inplace=True)
data.drop_duplicates(subset='username',keep='first',inplace=True)
count = pd.DataFrame()
count = data.iloc[:20,:]
sns.barplot(x=data.user_id,y=data.username)
plt.show()


# # Unigram

# In[80]:


def unigram(data):
    text = " ".join(data)
    CleanedText = re.sub(r'[^a-zA-Z]'," ",text)
    CleanedText = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(CleanedText) if word not in stopwords.words("indonesian") and len(word) > 3])
    return CleanedText


# In[87]:


def ngrams(data,n):
    text = " ".join(data)
    text1 = text.lower()
    text2 = re.sub(r'[^a-zA-Z]'," ",text1)
    text3 = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(text2) if word not in stopwords.words("english") and len(word) > 2])
    words = nltk.word_tokenize(text3)
    ngram = list(nltk.ngrams(words,n))
    return ngram


# In[161]:


(pd.Series(nltk.ngrams(words, 1)).value_counts())[:10]


# In[164]:


unigrams_series = (pd.Series(nltk.ngrams(words, 1)).value_counts())[:10]
unigrams_series.sort_values().plot.barh(color='red', width=.9, figsize=(12, 8))


# In[81]:


CleanedText = unigram(data['text'])


# In[86]:


from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
wordcloud = WordCloud(random_state=21).generate(CleanedText)
plt.figure(figsize = (10,5))
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()


# # Bigram

# In[92]:


Bigram_Freq = nltk.FreqDist(ngram)


# In[154]:


(pd.Series(nltk.ngrams(words, 2)).value_counts())[:10]


# In[165]:


bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:10]
bigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))


# In[93]:


bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
plt.figure(figsize = (10,5))
plt.imshow(bigram_wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()


# # Trigram

# In[96]:


Trigram_Freq = nltk.FreqDist(ngram)


# In[158]:


(pd.Series(nltk.ngrams(words, 3)).value_counts())[:10]


# In[166]:


trigrams_series = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:10]
trigrams_series.sort_values().plot.barh(color='green', width=.9, figsize=(12, 8))


# In[159]:


trigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Trigram_Freq)
plt.figure(figsize = (10,5))
plt.imshow(trigram_wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()


# # 4. Vektorisasi menggunakan teori TF dan TFIDF

# # Teori TF

# TF adalah Term Frequency t,dengan f(t,d) adalah pencacahan mentah istilah dalam dokumen, yaitu jumlah kemunculan istilah t dalam dokumen d. Semakin sering suatu istilah muncul, semakin besar nilai tf-nya.

# In[14]:


dataFinal = data[['text', 'cleaned_text','Label']]
modelSVC = mdg.modelling(data = dataFinal, modelname= '202106',                         crossval = False,  termfrequency = True,                              n_fold = 3, kernel = 'linear', n_jobs=1)


# # TFIDF

# Inversi frekuensi dokumen, idf(t, D), adalah ukuran informasi yang diberikan oleh istilah t, yaitu seberapa sering atau jarang sebuah istilah muncul dalam seluruh dokumen. Semakin jarang suatu istilah di antara dokumen, semakin besar nilai idf-nya. Nilainya adalah logaritma dari kebalikan dari jumlah dokumen yang memiliki istilah t yang dibagi jumlah seluruh dokumen (N), dengan himpunan {d ∈ D: t ∈ d} adalah himpunan dokumen d dalam D yang memiliki istilah t. 

# In[15]:


dataFinal = data[['text', 'cleaned_text','Label']]
modelSVC = mdg.modelling(data = dataFinal, modelname= '202106',                         crossval = False,  termfrequency = False,                              n_fold = 3, kernel = 'linear', n_jobs=1)


# # Analisis hasil evaluasi

# Berdasarkan kedua evaluasi, terlihat bahwa hasil modellingnya berbeda. Parameter C pada teori TF kurang bagus dibanding pada teori TFIDF. Terlihat juga pada hasil lainnya bahwa dengan menggunakan teori TF menghasilkan hasil yang sedikit berbeda dengan teori TFIDF. Hal ini dapat dikatakan jika hasil modelling akan mendapat hasil yang bagus jika menggunakan teori TFIDF.

# # 5. Support Vector Machine (Hard Marging & Soft Margin)

# SVM adalah sistem pembelajaran menggunakan ruang berupa fungsi – fungsi linear dalam sebuah ruang fitur yang berdimensi tinggi yang dilatih menggunakan algoritma pembelajaran berdasarkan pada teori optimasi dengan mengimplementasikan learning bias (Santosa, 2007). SVM memiliki prinsip dasar linier classifier yaitu kasus klasifikasi yang secara linier dapat dipisahkan, namun SVM telah dikembangkan agar dapat bekerja pada problem non-linier dengan memasukkan konsep kernel pada ruang kerja berdimensi tinggi (Pusphita Anna Octaviani, dkk, 2014).
# 
#    Support Vector Machine (SVM) menggunakan model linear sebagai decision boundary dengan bentuk umum sbb: 
#                      
#                      y(x) = wTf(x) + b
# 
# dimana x adalah vektor input, w adalah parameter bobot,
# f(x) adalah fungsi basis, dan b adalah suatu bias.
# 
# Bentuk model linear yang paling sederhana untuk decision boundary adalah:
#                        
#                      y(x) = wTx + w0
#                        
# Dimana x adalah vektor input, w adalah vektor bobot dan w0 adalah bias.
# 
#    Decision boundary adalah ketika y(x)=0, yaitu suatu hyperplane berdimensi (D-1). Untuk menentukan decision boundary (DB), yaitu suatu model linear atau hyperplane y(x) dengan parameter w dan b, SVM menggunakan konsep margin yang didefiniskan sebagai jarak terdekat antara DB dengan sembarang data training. Dengan memaksimalkan margin maka akan di dapat suatu DB tertentu.
# 
#    Margin merupakan jarak antara hyperplane dengan data terdekat dari masing – masing kelas. SVM akan mencari hyperplane terbaik yang berfungsi sebagai pemisah dua buah kelas pada ruang input. Hyperplane tersebut dapat berupa line pada two dimension dan dapat berupa flat plane pada multiple plane. 
# 
#    Hard – Margin SVM / Linear SVM yaitu Teknik SVM dimana merupakan klasifier yang menemukan hyperplane dengan kasus data yang digunakan merupakan data dengan dua kelas yang sudah terpisah secara linear. Margin maksimum dapat diperoleh dengan cara memaksimalkan nilai jarak antara hyperplane dan titik terdekatnya yaitu 1/||w||.
#     
#    Soft – Margin SVM Ketika data yang digunakan tidak sepenuhnya dapat dipisahkan, slack variables xi diperkenalkan kedalam fungsi obyektif SVM untuk memungkinkan kesalahan dalam misklasifikasi. Dalam hal ini, SVM bukan lagi hard margin classifier yang akan mengklasifikasi semua data dengan sempurna melainkan sebaliknya yaitu SVM soft margin classifier dengan mengklasifikasikan sebagian besar data dengan benar, sementara memungkinkan model untuk membuat misklasifikasi beberapa titik di sekitar batas pemisah.
#     
#    Persamaan soft margin hampir mirip dengan hard margin hanya terdapat sedikit modifikasi dengan adanya slack variabel

# In[120]:


modelSVC = mdg.modelling(data = dataFinal, modelname= '202106',                         crossval = False,  termfrequency = False,                              n_fold = 3, kernel = 'linear', n_jobs=1)


# # 6. Model Prediksi

# In[170]:


teks1 = 'Haha biang bangsat! Cebong lu!'


# In[173]:


mo.hateSpeechPredict(teks1)


# In[174]:


teks2 = ' Si bangsat cebong penyebar hoax ternyata anak banteng'


# In[175]:


mo.hateSpeechPredict(teks2)


# In[176]:


teks3 = 'Flu babi muncul di Cina, dan flu burung menyebar dr Cina ke dunia,sekarang virus korona telah muncul d Cina Masyarakat Cina adalah masyarakat yg korup secara mental. Kt harus menyelamatkan anak2 Tiongkok telah menjadi ancaman bagi kemanusiaan.#Corono'


# In[177]:


mo.hateSpeechPredict(teks3)


# In[178]:


teks4 = 'Bangsat cina. I said what i said. Fuck you sepet'


# In[179]:


mo.hateSpeechPredict(teks4)


# In[180]:


teks5 = 'Dasar komunis bangsat, cina bangsat, gara2 itu laga Arsenal vs city jg nngak di siarin di negerinya segitu bencinya sama Ozil yg mengucapkan rasa simpatinya..'


# In[181]:


mo.hateSpeechPredict(teks5)


# In[182]:


teks6 = 'WOI blantik ASU jelasin sejarah Cina setau saya bangsat cina yang menggarong di Indonesia #JanganPercayaBoneka #JanganPercayaBoneka'


# In[183]:


mo.hateSpeechPredict(teks6)


# In[184]:


teks7 = 'D manakah bangsat cina yg mmpjuangkan hak sama rata ketika ini? Maaf bukan aku rasis tp ini juga knyataan...oh ya,bukit kepong jgn lupa'


# In[185]:


mo.hateSpeechPredict(teks7)


# In[186]:


teks8 = ' Cita-cita kok meniduri 100 perempuan. Cita-cita tuh melindungi segenap bangsa Indonesia dan seluruh tumpah darah Indonesia, memajukan kesejahteraan umum, mencerdaskan kehidupan bangsa, dan ikut melaksanakan ketertiban dunia yang berdasarkan kemerdekaan'


# In[187]:


mo.hateSpeechPredict(teks8)


# In[188]:


teks9 = 'Sejak proklamasi kemerdekaan, Pancasila ditetapkan sebagai ideologi bangsa Indonesia, sebagai pedoman hidup masyarakat Indonesia dalam ber.....https://facebook.com/103754998279685/posts/198032845518566/…'


# In[189]:


mo.hateSpeechPredict(teks9)


# In[190]:


teks10 = 'SAYA PRIHATIN SEKALI GENERASI PENERUS BANGSA INDONESIA SEPERTI INI AKIBAT PEMBELAJARAN TATAP MUKA ON LINE YA DI BATASI MIN SEMINGGU 2X DAN HANYA 2 JAM'


# In[191]:


mo.hateSpeechPredict(teks10)


# # 7. Analisis Model

# Model SVM yang digunakan mendapatkan hasil modelling yang cukup baik pada database ini. Hasil yang diperoleh adalah sebagai berikut :
# 1. roc_auc model terbaik adalah: 0.9998816257195341;
# 2. roc_auc model estimator terbaik adalah: 0.9958236658932714;
# 3. Parameter terbaik adalah: {'svc__C': 0.1};
# 4. Rataan roc_auc model tiap fold adalah: 0.9997992133693031
# 
# Untuk kelebihan dari model yang terbentuk adalah waktu prosesnya yang cepat pada database ini, dan menghasilkan nilai prediksi yang mendekati nilai awal, juga mudah untuk diimplementasikan hasilnya. Namun kekurangan pada model ini secara umum adalah sulit dipakai dalam problem berskala besar, dalam arti jika datanya banyak maka akan mendapat kesulitan dalam prosesnya, bisa dikarenakan waktu yang digunakan cukup memakan waktu.

# In[ ]:




