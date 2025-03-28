import spacy
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from flask import Flask, request, abort
import community.community_louvain as community


#  Creating a flask app object
app = Flask(__name__)


#  Creating an API endpoint 'app route' that will take 
#  text recieve text file from user side and start the
#  processing.
@app.route("/network" , methods = ['POST'])
def get_network():
    if request.method == 'POST':

        #  Return a error 400 indicating bad request. And the user must input a text file
        if 'file' not in request.files:
            abort(400, description = "Please input a text file.")
        
        #  Get the name of the uploaded file.
        fileName = request.files['file'].filename

        #  Check if the file extension is .txt. If not then return an error.
        if not fileName.endswith('.txt'):
            abort(400, description = "Please upload a .txt file.")
        
        #  Else if the text file exists.
        else:
            Text_File = request.files['file'] 
            book = Text_File.read().decode("utf-8")
            processed_book = nlp(book)

            #  Creating a list containing dictionary objects with "sentence" and "entities" key-value pairs.
            sentence_entity_df = []
            for sentence in processed_book.sents:
                entities_list = [entity.text for entity in sentence.ents if entity.label_ == "PERSON"]
                sentence_entity_df.append({"sentence" : sentence , "entities" : entities_list})

            sentence_entity_df = pd.dataframe(sentence_entity_df)

            #  Creating a sentence and entities JSON

            
            

    return





