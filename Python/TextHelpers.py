import itertools
import re
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
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

def remove_stopwords(X):   
    """
    Removes unnecessary words from dataset
    """
    lemmatizer = WordNetLemmatizer()
    questions = []
    stopWords = set(stopwords.words('english'))
    for question in X:
        wordTokens = word_tokenize(question)
        filteredSentence = [lemmatizer.lemmatize(w) for w in wordTokens if not w in stopWords]
        questions.append(' '.join(filteredSentence))
    return questions

def cleanInput(string):
    """
	removing whitespaces,useless characters from dataset.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def createVocabulary(X):
    
    '''
        Creates a vocabulary of words and maps them to an index.
    '''
    word2index = {}
    vocabularySet = Counter()
    vocabularyFile = open("vocabulary.txt","w")
    english_words = set(words.words())
    for questions in X:
        questionsWords = questions.split(' ')
        for word in questionsWords:
            word = cleanInput(word.lower())
            if(word in english_words):
                vocabularySet[word] = vocabularySet[word]+1

    for index,word in enumerate(vocabularySet):
        word2index[word] = index
        vocabularyFile.write(word + "\n")
        
    return [word2index, vocabularySet, vocabularyFile]   

def featureExtractionFrequency(X, vocabularySet, word2index):
    '''
        Creates word vectors using frequency based representation
    '''

    if(len(word2index)!= len(vocabularySet)):
        print("Error!")
        return -1
    lengthOfVector = len(vocabularySet)    
    featureDataset = []
    for question in X:
        newFeatureVector = np.zeros(lengthOfVector, dtype = float)

        for word in question:
            word = re.sub(r"\b[a-zA-Z]\b", "",word)
            if(word.lower() in word2index.keys()):
                newFeatureVector[word2index[word.lower()]]+=1
        featureDataset.append(newFeatureVector)
    return [np.array(featureDataset), lengthOfVector]    

def featureExtactionUsingIndex(X, word2index, vocabularySet):
  if(len(word2index)!=len(vocabularySet)):
    return -1
  lengthOfVector = len(vocabularySet)
  maxLength = max([len(sentence.split(' ')) for sentence in X])
  featureDataset = []
  for question in X:
    increment = 0
    newFeatureVector = np.zeros(lengthOfVector, dtype = float)
    for word in question:
      if(word in word2index.keys()):
        newFeatureVector[increment] = word2index[word]
        increment+=1
    featureDataset(newFeatureVector)
  return [np.array(featureDataset), lengthOfVector]

def callFunctions():
  X,Y = readData()   
  X = remove_stopwords(X) 
  word2index, vocabularySet, vocabularyFile = createVocabulary(X)
  X, lengthOfVector = featureExtractionFrequency(X, vocabularySet, word2index)
  print(X.shape)
  return [X,Y,lengthOfVector]

callFunctions()