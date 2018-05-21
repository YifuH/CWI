# -*- coding: utf-8 -*-


from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from nltk.corpus import wordnet

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        #self.model = RandomForestClassifier(n_estimators = 100) 
        self.model = svm.SVR()
        #self.model = tree.DecisionTreeClassifier()
    
    def count_synonyms(self, dictionary, word):
        for 
        
    
    #build bigrams and trigrams 
    def bi_tri_gram(self, word):
        word = word.lower()
        bi_char=" "
        bi_char_list=[]
        tri_char=" "
        tri_char_list=[]
        for char in word:
            bi_char+=char
            bi_char_list.append(bi_char)
            bi_char=""
            bi_char+=char
        bi_char+=" "
        bi_char_list.append(bi_char)
        
        for index in range(len(word)-1):
            tri_char+=word[index]
            tri_char+=word[index+1]
            tri_char_list.append(tri_char)
            tri_char=""
            tri_char+=word[index]
        
        tri_char+=word[-1]
        tri_char+=" "
        tri_char_list.append(tri_char)
            
        
        return (bi_char_list, tri_char_list)
        
    def extract_features(self, word):
        word = word.lower()
        dict_feature = {}
        #dict_feature["len_chars"] = len(word) / self.avg_word_length        
        
        bi_c_l, tri_c_l =self.bi_tri_gram(word)
        for i in bi_c_l:
            if i in self.w_bi:
                dict_feature[i]=self.w_bi[i]
                
        for i in tri_c_l:
            if i in self.w_tri:
                dict_feature[i]=self.w_tri[i]

        return dict_feature

    def build_dict_vector(self,trainset):
        dict_vector = {} 

        # build all features
        for sent in trainset:
            dict_one = self.extract_features(sent['target_word'])
            dict_vector = dict(dict_vector, **dict_one)

        feature_vectors = DictVectorizer()
        feature_vectors.fit_transform(dict_vector)
        
        return feature_vectors

    def train(self, trainset):
        self.w_bi = {}
        self.w_tri = {}
        #get the number of all words
        #tokenstr = nltk.word_tokenize(trainset.lower())
        for sent in trainset:
            bi_c_l, tri_c_l =self.bi_tri_gram(sent['target_word'])
            for i in bi_c_l:
                if i in self.w_bi:
                    self.w_bi[i]+=1
                else:
                    self.w_bi[i]=1
            for i in tri_c_l:
                if i in self.w_tri:
                    self.w_tri[i]+=1
                else:
                    self.w_tri[i]=1               
            #word frequency
            #self.fre_dist = nltk.FreqDist(tokenstr)
                
        self.all_feature_vectors = self.build_dict_vector(trainset)
        X = np.zeros((len(trainset), len(self.all_feature_vectors.vocabulary_)))
        y = []
 
        
        for i in range(len(trainset)):
            X[i,:] = np.hstack([self.all_feature_vectors.transform(self.
             extract_features(trainset[i]['target_word'])).A[0]])
            y.append(trainset[i]['gold_label'])

        self.model.fit(X, y)

    def test(self, testset):
            
            
        X = np.zeros((len(testset), len(self.all_feature_vectors.vocabulary_)))
        for i in range(len(testset)):
            #if i %200 == 0: print ("\rtested... %d" %i, end = '\r')

            x_features = self.extract_features(testset[i]['target_word'])

            x = self.all_feature_vectors.transform(x_features).A[0]

            X[i,:] = x

        return self.model.predict(X)
