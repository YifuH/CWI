# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:51:15 2018

@author: HPC
"""

from sklearn.ensemble import RandomForestClassifier
    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2
            
        self.model = RandomForestClassifier(n_estimators = 100, min_samples_split = 5,
                                             min_samples_leaf= 1, max_features = 'auto', max_depth = 110,
                                             bootstrap = False,random_state = 5) 
        
        

import torch.nn as nn        
            def __init__(self):
        self.embedding_model = spacy.load('en_core_web_lg')
        self.h_en = Hyphenator('en_US')

    def word_embeddings(self,target_word):
        embeddings = self.embedding_model(target_word)
        embeddings = torch.unsqueeze(torch.FloatTensor(embeddings.vector),dim=0)
        return embeddings   # 1 * 300
    
    























