# -*- coding: utf-8 -*-


from sklearn.linear_model import LogisticRegression
import random

random.seed(1)
R = random.random()

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2
            
        self.w={}
        self.model = LogisticRegression()
        
    def tri_nn_train(self,data,y):
        Iteration=100
        
        
        random.shuffle(data, lambda: R)
        for listc in data:
            for tri_char in listc:
                self.w[tri_char]=0
        
        direction=1
        for i in range(Iteration):
            for listc in data:              
                y_train=0
                for tri_char in listc:
                    y_train+=self.w[tri_char]
                if y_train > 0:
                    if y ==1:  #right
                        direction=1
                    else:
                        direction=-1
                else:
                    if y ==1:  
                        direction=-1
                    else:
                        direction=1
                        
                for tri_char in listc:
                    self.w[tri_char]+=direction

    
    def tri_nn_predict(self,data):
        
        y_list=[]
        for listc in data:
            y_predict=0
            for tri_char in listc:
                if tri_char not in self.w:
                    y_predict+=1
                else:
                    y_predict+=self.w[tri_char]
            if y_predict > 0:
                y_list.append("1")
            else:
                y_list.append("0")

        return y_list

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        
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
            
        
        return tri_char_list
#        return [len_chars, len_tokens]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.tri_nn_train(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.tri_nn_predict(X)
