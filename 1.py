from tqdm.auto import tqdm
from time import sleep


import gensim
from gensim import corpora

import pickle
import spacy
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])
import re


def set_clean(text):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['disabled', 'color', 'default', 'composite', 'screen', 'nwcr', 'naval', 'college', 'review', 'volume', 'winter', 'spring', 'common', 'digital', 'many', 'time', 'book', 'year', 'first', 'issue', 'issue', 'number', 'profile', 'group', 'large', 'article', 'follow', 'additional', 'work', 'common', 'edu', 'nwc', 'review', 'recommend', 'citation', 'page' ])
    set_stop_words = set(stop_words)
    return [w for w in text if w.lower() not in set_stop_words]

def prep_clean(text):
    clean_txt = []
    for i in tqdm(text):
        word = i.strip('][').split(', ')       
        for j in tqdm(word):
            sleep(0.0000000000001)
            k = re.sub("'", '', j)
            clean_txt.append(k)

    zzz = set_clean(clean_txt)

    return zzz

def lemmatization(texts,allowed_postags=['NOUN', 'ADJ']): 
    output = []
    for sent in tqdm(texts):
        sleep(0.0000000000001)
        doc = nlp(sent) 
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
    return output


with open(r'corpus.pkl', 'rb') as q:
    review_data_00_05, review_data_06_10, review_data_11_15, review_data_16_20  = pickle.load(q)   

    text_list_00_05=review_data_00_05['Data'].tolist()
    text_list_06_10=review_data_06_10['Data'].tolist()
    text_list_11_15=review_data_11_15['Data'].tolist()
    text_list_16_20=review_data_16_20['Data'].tolist()

    text_list_rem_00_05 = prep_clean(text_list_00_05)
    text_list_rem_06_10 = prep_clean(text_list_06_10)
    text_list_rem_11_15 = prep_clean(text_list_11_15)
    text_list_rem_16_20 = prep_clean(text_list_16_20)
    print("text_list_rem_16_20 = prep_clean(text_list_16_20)")
    tokenized_reviews_00_05 = lemmatization(text_list_rem_00_05)
    tokenized_reviews_06_10 = lemmatization(text_list_rem_06_10)
    tokenized_reviews_11_15 = lemmatization(text_list_rem_11_15)
    tokenized_reviews_16_20 = lemmatization(text_list_rem_16_20)
    print("tokenized_reviews_16_20 = lemmatization(text_list_rem_16_20)")
    dictionary_00 = corpora.Dictionary(tokenized_reviews_00_05)
    dictionary_06 = corpora.Dictionary(tokenized_reviews_06_10)
    dictionary_11 = corpora.Dictionary(tokenized_reviews_11_15)
    dictionary_16 = corpora.Dictionary(tokenized_reviews_16_20)
    print("    dictionary_16 = corpora.Dictionary(tokenized_reviews_16_20)")
    doc_term_matrix_00 = [dictionary_00.doc2bow(rev) for rev in tokenized_reviews_00_05]
    doc_term_matrix_06 = [dictionary_06.doc2bow(rev) for rev in tokenized_reviews_06_10]
    doc_term_matrix_11 = [dictionary_11.doc2bow(rev) for rev in tokenized_reviews_11_15]
    doc_term_matrix_16 = [dictionary_16.doc2bow(rev) for rev in tokenized_reviews_16_20]

    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamulticore

    print("Build LDA model1")
    lda_model_00 = LDA(corpus=doc_term_matrix_00, id2word=dictionary_00, num_topics=10, random_state=100,
                chunksize=1000, passes=50,iterations=100)
    print("Build LDA model2")
    lda_model_06 = LDA(corpus=doc_term_matrix_06, id2word=dictionary_06, num_topics=10, random_state=100,
                chunksize=1000, passes=50,iterations=100)
    print("Build LDA model3")
    lda_model_11 = LDA(corpus=doc_term_matrix_11, id2word=dictionary_11, num_topics=10, random_state=100,
                chunksize=1000, passes=50,iterations=100)
    print("Build LDA model4")
    lda_model_16 = LDA(corpus=doc_term_matrix_16, id2word=dictionary_16, num_topics=10, random_state=100,
                chunksize=1000, passes=50,iterations=100)


    file = open('done.pkl', 'wb')
    pickle.dump((lda_model_00,lda_model_06,lda_model_11,lda_model_16, doc_term_matrix_00, doc_term_matrix_06, doc_term_matrix_11, doc_term_matrix_16, dictionary_00, dictionary_06, dictionary_11, dictionary_16), file)
    file.close()