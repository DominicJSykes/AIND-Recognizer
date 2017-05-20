import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    #DS - Implemented the recognizer function which selects the most likely word based on the models previously selected.
    
    #Create variables to store probabilites that the test word is represented by the model, and the best guess for each 
    #test word.
    probabilities = []
    guesses = []
    
    #Iterate over each test instance.
    for test, test_lengths in test_set.get_all_Xlengths().values():
        
        #Create variables to store current best word, best score and probabilites for each model.
        best_score = float("-inf")
        best_word = None
        item_probabilities = {}
        
        #Iterate over each model
        for word in models:
            try:
                #Calculate log likelihood of the model generating the test instance.
                logL = models[word].score(test,test_lengths)
                
                #If the score is better than the current best, update best score and best word match
                if logL > best_score:
                    best_score = logL
                    best_word = word
            except:
                #If calculation throws an error return minimum log likelihood.
                logL = float("-inf")
                
            #Store probability for each word
            item_probabilities[word] = logL
        
        #Store probabilities of each word for each test instance, and the best guess for each test instance
        probabilities.append(item_probabilities)
        guesses.append(best_word)
        
    return (probabilities, guesses)
