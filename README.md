# News Headline Classification into Genres
## Overview
This project aims to use a pretrained model and fine tune it to predict the genre of a specific news headline
## Details
This project makes use of the Bert classification model and uses the publicly available AG News dataset to fine tune the model. It then uses the 
Gradio interface to launch a GUI based interface where users can enter in headlines and get back the predicted genre out of 4 categories which are-
1. World
2. Sports
3. Business
4. Science/Technology\
Any flagged predictions may get stored in the flagged subfolder which is formed after the interface.py script is run for the first time.
## Steps to run and try the model
1. Download the folder from Google Drive and make sure it contains-
    1. Training Script
    2. train.csv
    3. A subfolder which has a json and a bin file
    4. interface.py
    5. This README file
2. Extract all of the above with the same directory structure
3. This model is already fine tuned and hence does not require retraining. Just run interface.py and upon prompt open the link in the web browser
4. The gradio application would open and provide results
5. Take care not to leave the process running after closing the Gradio app.
## Steps to get missing files
https://drive.google.com/drive/folders/1j3wEYFqKvq9Wp4535byjXajp4OEfv245?usp=sharing
\Access the above link to download train.csv and the folder containing the model

### Author
Arnav Garg\
22B1021
