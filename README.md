<h1 align="center">Disaster Response Pipeline</h1>

<h3 align="center">Udacity Data Scientist Nanodegree Program Project 2</h3>

![image](https://github.com/hviher/Udacity-Disaster-Response-Pipeline/blob/main/Intro_screenshot.JPG)

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [File Descriptions](#file_description)
4. [Instructions](#instructions)
5. [Web App](#webapp)
6. [Licensing, Authors, Acknowledgments](#licensing)

## Introduction <a name="introduction"></a>
This project is part of Udacity's Data Scientist Nanodegree Program.

Pre-labelled real messages that were sent during disaster events were provided by [Figure Eight](https://www.figure-eight.com/).  These messages will be used to create a disaster response model that will categorize real time messages in order to send the messages to the appropriate disaster relief agency.

The project will include a web app where emergency workers can input a new message and get classification results in several categories.

## Installation <a name="installation"></a>
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## File Descriptions <a name="file_description"></a>

### main directory:<br/>
- **ETL Pipeline Preparation.ipynb** - Jupyter Notebook with step by step process used to help build process_data.py
- **ML Pipeline Preparation.ipynb**- Jupyter Notebook with step by step process used to help build train_classifier.py

### app folder:<br/>
- **run.py** - python script to run the web app<br/>
- **app/templates folder** - html template files for web app<br/>

### data folder:<br/>
- **disaster_messages.csv** - data containing real messages that were sent during disaster events<br/>
- **disaster_categories.csv** - data containing categories associated with the messages<br/>
- **DisasterResponse.db** - cleaned SQLite database<br/>
- **process_data.py** - python script to run ETL pipeline that cleans data and stores in database<br/>

### models folder:<br/>
- **train_classifier.py** - python script to run ML pipeline that trains classifier and saves the trained model<br/>

## Instructions <a name="instructions"></a>

To develop the disaster response model, run the following command lines:

1. **python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db**  - loads, cleans and saves the data in an SQLite database 

2. **python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl** - loads data, trains classifier, runs a grid search and saves the classifier as a pickle file
 
3. **python run.py** - launches the web app. 

4. Go to http://0.0.0.0:3001/
 
## Web App <a name="webapp"></a>

Once the web app has launched, you will see on the main page a place to enter and submit a message which will return a classification for the message. 

As well, a couple of visualizations are displayed on the data.
<p align="center">
  <img src="https://github.com/hviher/Udacity-Disaster-Response-Pipeline/blob/main/ScreenShot1.JPG" />
</p>

In this screenshot you can see the categories from submitting a message ('need food'):

<p align="center">
  <img src="https://github.com/hviher/Udacity-Disaster-Response-Pipeline/blob/main/ScreenShot2.JPG" /)
</p>

## Licensing, Authors, Acknowledgments <a name="licensing"></a>
 
I would like to thank [Figure Eight](https://www.figure-eight.com/) for providing the data to complete this project. 
The licensing can be found [here](https://github.com/hviher/Udacity-Disaster-Response-Pipeline/blob/main/LICENSE).
Otherwise, feel free to use the code here as you would like!
