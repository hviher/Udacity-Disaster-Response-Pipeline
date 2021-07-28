import sys
import warnings
warnings.filterwarnings('ignore')

# import libraries
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'ignore'])
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    """Loads the database created from process_data.py

    Args:
        database_filepath - path for database

    Returns:
        Target variables X and y 

    """
    
    # load data from database with read_sql_table    
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterData', con = engine)
    
    #Define feature and target variables X and Y
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)    
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """Clean and tokenize the text data
    
    Args: 
        text - text data that needs to be cleaned and tokenized
    
    Returns: 
        clean_tokens - list of tokens extracted from the text data
    """
    
    #regular expression to detect a url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # normalize text    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:     
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds a machine learning pipeline
    
    Args: 
        none
    
    Returns: 
        cv - model produced from grid search
    """    
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #specify parameters for grid search
    parameters = {'clf__estimator__min_samples_split': [3, 4],
                  'clf__estimator__n_estimators': [20, 40]}    

    # create a grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """Evaluates the machine learning model
    
    Args: 
        model - ML model from build_model using grid search
        X_test - messages
        y_test - categories
        category_names - category names associated with y_test
        
    Returns: 
        prints out f1 score, precision and recall for each category in the dataset
    """  
    
    # testing the model
    y_pred_test = model.predict(X_test)

    # Print the report on f1 score, precision and recall for each output category
    print(classification_report(y_test.values, y_pred_test, target_names = category_names))

def save_model(model, model_filepath):
    """Exports the model as a pickle file
    
    Args: 
        model - ML model from build_model using grid search
        model_filepath - messages
        
    Returns: 
        none
    """
    # export the model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()