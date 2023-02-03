import sys
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import WordPunctTokenizer
import nltk
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

nltk.download('stopwords')
warnings.filterwarnings('ignore')

#Source: https://www.kaggle.com/jameelkaggle/yelp-cf-recommender
def init_ds(json):

    
    ds= {}
    keys = json.keys()
    for k in keys:
        ds[k]= []
    return ds, keys

def read_json(file):
    dataset = {}
    keys = []
    with open(file, encoding="utf8") as file_lines:
        for count, line in enumerate(file_lines):
            data = json.loads(line.strip())
            if count ==0:
                dataset, keys = init_ds(data)
            for k in keys:
                dataset[k].append(data[k])
                
        return pd.DataFrame(dataset)

#https://github.com/gann0001/Restaurant-Recommendation-System
#Cleans up the text, removing characters that match regex
def clean_text(text):
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)    
    return text

#Preprocess original data set, this is not needed after running once unless new dataset gets deleted
#Uses https://towardsdatascience.com/converting-yelp-dataset-to-csv-using-pandas-2a4c8f03bd88
#To help preprocess and create csv file
def preprocess():
    print("Hello")

    city = "toronto"
    #Business
    business_json_path = 'data/business.json'

    df_b = read_json(business_json_path)
    df_b = df_b[df_b['is_open']==1]

    drop_columns = ['hours','is_open','review_count', 'latitude', 'longitude', 'attributes', 'hours', 'postal_code', 'address']
    df_b = df_b.drop(drop_columns, axis=1)

    business_Bars = df_b[df_b['categories'].str.match('Bars',case=True, na=False)]
    toronta_Bars = business_Bars[business_Bars['city'].str.match('Toronto',case=True, na=False)]

    review_json_path = 'data/review.json'

    size = 10000
    review = pd.read_json(review_json_path, lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                      chunksize=size)

    chunk_list = []
    for chunk_review in review:
        # Drop columns that aren't needed
        chunk_review = chunk_review.drop(['review_id','useful','funny','cool'], axis=1)
        # Renaming column name to avoid conflict with business overall star rating
        chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
        # Inner merge with edited business file so only reviews related to the business remain
        chunk_merged = pd.merge(toronta_Bars, chunk_review, on='business_id', how='inner')
        
        chunk_list.append(chunk_merged)
    # After trimming down the review file, concatenate all relevant data back to one dataframe
    df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)

    #Data only within time span
    start_date = '01-01-2015 00:00:00'
    end_date = '01-01-2019 00:00:00'
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] > start_date) & (df['date'] < end_date)]

    csv_name = "Data/bars.csv"
    df.to_csv(csv_name, index=False)


def menu():
    my_df = pd.read_csv('Data/bars.csv')
    temp = my_df.drop_duplicates(subset='name')
    print("************Recommender System**************")
    print()

    st = "Enter A Bar Number Between 0 and " + str(temp['name'].count()-1) + ": "
    #Find the name of the bar
    select = int(input(st))
    choice = temp.iloc[[select]]['name']
    cho = choice.to_string(index=False)
    print("Selection:", cho)

    #choice = choice['name']

    #counter = 0
    #f = open("Data/barIndexNames.txt", "w")
    #for i in temp['name']:
        #title = i
        #store = str(counter) + ". " + title + '\n'
        #counter += 1
        #f.write(store)

    #print("The Bar You Have Chosen Is:", cho)

    #print()
    #print("********************************************")

    #Shows number of distinct business names
    #print("Unique Names:", my_df['name'].nunique())

    return cho, my_df

#Source: https://www.youtube.com/watch?v=_2nES58GEHM
def collaborative(selection, df_b):
    #my_df = pd.read_csv('bars.csv')
    #Creates a data frame that shows the name of the business along with the average rating that has been given
    ratings = pd.DataFrame(df_b.groupby('name')['review_stars'].mean())
    ratings['Number_of_Ratings'] = df_b.groupby('name')['review_stars'].count()


    #Create user item interaction matrix
    #Tells the relationship between individual users and individual bars
    bar_matrix_UII = df_b.pivot_table(index='user_id', columns='name', values='review_stars')


    #Sorting to see which has highest number of ratings
    #print(ratings.sort_values('Number_of_Ratings', ascending=False).head(10))

    #Making recommendation by find the user selection in the matrix
    selected_user_rating = bar_matrix_UII[selection]

    #Finds correlations with other bars
    similar = bar_matrix_UII.corrwith(selected_user_rating)


    #Creating a threshold for minimum number of ratings
    corr_similar = pd.DataFrame(similar, columns=['Correlation'])
    corr_similar.dropna(inplace=True)

    #Bring in ratings
    corr_similar = corr_similar.join(ratings['Number_of_Ratings'])

    test = corr_similar[corr_similar['Number_of_Ratings'] > 15].sort_values(by='Correlation', ascending=False).head(10)


    #Ignore selection in correlation
    CF_Bars = []
    for i in range(len(test)):
        item = test.index[i]
        if item != selection:
            CF_Bars.append(item)

    #Holds Recommended Bars from CF
    
    return CF_Bars

#Source: https://www.datacamp.com/community/tutorials/recommender-systems-python
def get_recommendations(data, indices, bID, cosine_sim):
    # Get the index of the bar that matches the title
    idx = indices[bID]

    # Get the pairwsie similarity scores of all bars with that bar
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the bars based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar bars
    sim_scores = sim_scores[1:11]

    # Get the bars indices
    bar_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar bars
    return data['business_id'].iloc[bar_indices]

#Source: https://www.datacamp.com/community/tutorials/recommender-systems-python
def content(df, selection):
    yelp_data = df[['business_id', 'user_id', 'stars', 'text']]
    yelp_data['text'] = yelp_data['text'].apply(clean_text)

    userid_df = yelp_data[['user_id','text']]
    business_df = yelp_data[['business_id', 'text']]

    #Join all review text for user and business
    userid_df = userid_df.groupby('user_id').agg({'text': ' '.join})
    business_df = business_df.groupby('business_id', as_index=False).agg({'text': ' '.join})

    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    business_df['text'] = business_df['text'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(business_df['text'])
    #Output the shape of tfidf_matrix
    tfidf_matrix.shape
    tfidf.get_feature_names()[5000:5010]

    #Distance similarities
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim.shape

    #Used to find bar
    indices = pd.Series(business_df.index, index=business_df['business_id']).drop_duplicates()

    #Gets first lookup
    id_select = df.loc[df.name==selection, 'business_id'].values[0]


    recoms = get_recommendations(business_df, indices, id_select, cosine_sim)

    CB = []
    for i in recoms:
        CB.append(df.loc[df.business_id==i, 'name'].values[0])

    return CB

def hybrid(collab, content, selection):
    if len(content) == 0:
        print("Collab:", collab)

    #Intersection of both candidates
    intersect = list(set(content) & set(collab))

    int_length = len(intersect)
    count = 1

    print("************Recommended Bars**************")
    print()

    if int_length >= 5:
        print("Top recommendations found using an intersection of CF and CB based on " + selection + ":")
        print()
        intersect = intersect[:5]
        for i in intersect:
            print(count, i)
            count += 1

    else:
        print("Top recommendations found using a union of CF and CB based on " + selection + ":")
        print()
        union = content + list(set(collab) - set(content))
        union = union[:5]
        for j in union:
            print(count, j)
            count += 1



def main():
    start = time.time()
    #preprocess()
    selection, df = menu()
    CF = collaborative(selection, df)
    CB = content(df, selection)
    hybrid(CF, CB, selection)
    print(f'Time: {time.time() - start}')

if __name__ == '__main__':
    main()
