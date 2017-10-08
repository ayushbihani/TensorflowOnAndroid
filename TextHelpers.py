import numpy as np 
import itertools
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''
Since labels are in text, we would want to convert them into a numeric format.
DESC : Description
ENTY : Entity
HUM : Human
NUM :  Numeric
LOC : Location
'''

label_map = {"DESC":0, "ENTY":1, "HUM":2, "NUM":3, "LOC":4, "ABBR":5}

def readData():
    questions=[]
    labels=[]

    with open('data.txt') as f:
        for line in f.readlines():
            data = line.split(':') #Split data into label and question
            labels.append(label_map[data[0]])
            questions.append(data[1])
    
    f.close()
    return [questions, labels]            

def remove_stopwords(data):
    
    """
    Removes unnecessary words from dataset
    """

    lemmatizer = WordNetLemmatizer()
    questions = []
    stop_words = set(stopwords.words('english'))
    for i in data:
        word_tokens = word_tokenize(i)
        filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w in stop_words]
        questions.append(' '.join(filtered_sentence))

    print(questions)
    return questions

X,y = readData()   
remove_stopwords(X)