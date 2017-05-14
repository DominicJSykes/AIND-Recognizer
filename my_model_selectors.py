import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    
    """where L is the likelihood of the fitted model, p is the number of parameters, and N is the number of data
    points."""

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        best_score = float("inf")
        best_model = None    
        for n in range(self.min_n_components,self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=n,n_iter=1000,random_state=self.random_state).fit(self.X,self.lengths)
                logL = model.score(self.X,self.lengths)
                logN = math.log(len(self.X))
                p = n * n + 2 * n * len(self.X[0]) - 1
                BIC = -2 * logL + p * logN
                if BIC < best_score:
                    best_model = model
                    best_score = BIC
            except:
                pass        
        return best_model
        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        best_score = float("-inf")
        best_model = None                     
        other_words = self.hwords.copy()
        del other_words[self.this_word]
        words = [self.hwords[word] for word in other_words]
        for n in range(self.min_n_components,self.max_n_components + 1):
            antiLogL = 0.0
            wc = 0
            try:
                model = GaussianHMM(n_components=n,n_iter=1000,random_state=self.random_state).fit(self.X,self.lengths)
                logL = model.score(self.X,self.lengths)
            except:
                continue 
            p = n * n + 2 * n * len(self.X) - 1
            DIC =  logL - np.mean(sum([model.score(self.hwords[word][0],self.hwords[word][1]) 
                   for word in other_words]))
            if DIC > best_score:
                best_model = model
                best_score = DIC       
        return best_model
        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        best_score = float("-inf")
        best_model = None
        if len(self.sequences) < 3:
            split_method = KFold(n_splits=2)
        else:
            split_method = KFold()
        for n in range(self.min_n_components,self.max_n_components + 1):   
            score = 0
            count = 0
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test, test_lengths = combine_sequences(cv_test_idx, self.sequences)                      
                    try:
                        trained_model = (GaussianHMM(n_components=n,n_iter=1000,random_state=self.random_state)
                                        .fit(train,train_lengths))
                        score += trained_model.score(test,test_lengths)
                        count += 1
                    except:
                        pass 
                if count > 0:
                    avg_score = score / count
                    if avg_score > best_score:
                        best_model = GaussianHMM(n_components=n,n_iter=1000).fit(self.X,self.lengths)
                        best_score = avg_score
            except:
                pass
        return best_model
    
        # TODO implement model selection using CV
        raise NotImplementedError
