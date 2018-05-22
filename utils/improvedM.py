# -*- coding: utf-8 -*-


from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

from nltk.corpus import wordnet

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = RandomForestClassifier(n_estimators = 100) 
        #self.model = svm.SVC()
#        self.model = tree.DecisionTreeClassifier()
    
    #count the number of synonyms and hyponyms
    def count_syn_hyp(self, dictionary, word):
        word = word.lower().split(" ")
        num = 0
        num_2 = 0
        for i in word:
            #syn is a list of synonyms
            syn = wordnet.synsets(i)
            num += len(syn)
            if len(syn)>0:
                num_2 += len(syn[0].hyponyms())
        dictionary["num_syn"] = num
        dictionary["num_hyp"] = num_2
        
        return dictionary
    
    #build bigrams and trigrams 
    def bi_tri_gram(self, word):
        word = word.lower()
        bi_char=" "
        bi_char_list=[]
        tri_char=" "
        tri_char_list=[]
        #get all bigrams
        for char in word:
            bi_char+=char
            bi_char_list.append(bi_char)
            bi_char=""
            bi_char+=char
        bi_char+=" "
        bi_char_list.append(bi_char)
        #get all trigrams
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
        dict_feature["len_chars"] = len(word) / self.avg_word_length        
        # get the ngrams features
        bi_c_l, tri_c_l =self.bi_tri_gram(word)
        for i in bi_c_l:
            if i in self.w_bi:
                dict_feature[i]=self.w_bi[i]
                
        for i in tri_c_l:
            if i in self.w_tri:
                dict_feature[i]=self.w_tri[i]
        #for english, get the synonyms and hyponyms features
        if self.language == "english":
            dict_feature = self.count_syn_hyp(dict_feature, word)
        #return the dictinary
        return dict_feature

    def build_dict_vector(self,trainset):
        dict_vector = {} 

        # get all features at first
        for sent in trainset:
            dict_one = self.extract_features(sent['target_word'])
            dict_vector = dict(dict_vector, **dict_one)
        
        #build the vectors to save all features
        feature_vectors = DictVectorizer()
        feature_vectors.fit_transform(dict_vector)
        
        return feature_vectors

    def train(self, trainset):
        self.w_bi = {}
        self.w_tri = {}
        
        #create the dict fot ngrams
        for sent in trainset:
            bi_c_l, tri_c_l =self.bi_tri_gram(sent['target_word'])
            #use the frequency to present the ngrams
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
        #build the vectors     
        self.all_feature_vectors = self.build_dict_vector(trainset)
        X = np.zeros((len(trainset), len(self.all_feature_vectors.vocabulary_)))
        y = []
 
        #bulid the training and testing matrices
        for i in range(len(trainset)):
            X[i,:] = np.hstack([self.all_feature_vectors.transform(self.
             extract_features(trainset[i]['target_word'])).A[0]])
            y.append(trainset[i]['gold_label'])  
            
        self.model.fit(X, y)
        #plot the learning curve
        #self.plot_learningCurve(X,y)

    def test(self, testset):
            
            
        X = np.zeros((len(testset), len(self.all_feature_vectors.vocabulary_)))
        for i in range(len(testset)):
            x_features = self.extract_features(testset[i]['target_word'])
            x = self.all_feature_vectors.transform(x_features).A[0]
            X[i,:] = x
            
        return self.model.predict(X)
    
    #plot learning curve
    def plot_learningCurve(self,X,y):
        plt.figure()
        plt.title("Learning Curves--DecisionTreeClassifier ")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes = np.linspace(.1, 1.0,5)
        #get results
        train_sizes, train_scores,test_scores= learning_curve(self.model, X, y, n_jobs=1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="b")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
                 label="Cross-validation score")
        plt.legend(loc="best")
        plt.show()
        
        
        
        
        
        
        
        
