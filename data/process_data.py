import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the message and category csv files for the model

    Args:
        messages_filepath - path for the message data
        categories_filepath - path for the category data

    Returns:
        dataframe of merged message and category data

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on=['id'])
    
    return df


def clean_data(df):
    """Cleans the data for the model

    Args:
        df - dataframe containing the text messages and the categories
        
    Returns:
        df - dataframe containing cleaned data

    """
    # create a dataframe of individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()    
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop child_alone since it is all zeros
    df = df.drop('child_alone', axis = 1)
    
    #replace the 2's in 'related' with 1's - making the assumption they are errors
    df['related'] = df['related'].replace(2, 1)
        
    # drop duplicates
    df = df.drop_duplicates()

    return df
    
def save_data(df, database_filename):
    """Saves the clean dataframe into an sqlite database

    Args:
        df - dataframe containing cleaned data
        
    Returns:
        database_filename - sqlite database

    """  
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterData', engine, index=False, if_exists = 'replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()