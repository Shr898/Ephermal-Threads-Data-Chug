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

# Return a error 400 indicating bad request. The user must input a text file
        if 'file' not in request.files:
            abort(400, description = "Please input a text file.")

    return



if __name__ == '__main__':  
   app.run() 

