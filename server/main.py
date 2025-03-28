#Codebase to be removed...


import spacy as sp
import pandas as pd
import networkx as nx
from pyvis.network import Network                                         #Does the interactive visualizaion graph.
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import community.community_louvain as cl                                  #Does community detection.
import streamlit.components.v1 as components                              #Used to display the html pyvis-interactive visualization.


st.set_page_config(page_title="Data_Chug")

nlp = sp.load("en_core_web_sm")

st.text("Input a text-file to print the network")
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    book = uploaded_file.read().decode("utf-8")

    doc = nlp(book)

    # Creating a list that contains all the characters in the text.
    sent_ent_df = []
    for sent in doc.sents:
        entity_list = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
        sent_ent_df.append({"sentence": sent, "entities": entity_list})

    # converting sent_ent_df into a python dataframe
    sent_ent_df = pd.DataFrame(sent_ent_df)
    relations = []

    # removing all the rows of length <1
    sent_ent_df = sent_ent_df[sent_ent_df['entities'].apply(len) > 1]

    # making a relations list that contains source and target
    for i in range(sent_ent_df.index[-1]):
        end_i = min(i+5, sent_ent_df.index[-1])
        char_list = sum((sent_ent_df.loc[i: end_i].entities), [])

        char_unique = [char_list[i] for i in range(len(char_list))
                       if (i == 0) or char_list[i] != char_list[i-1]]

        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx+1]
                relations.append({"source": a, "target": b})

    # converting the relations list to a python dataframe
    relations = pd.DataFrame(relations)

    # sorting the relations dataframe
    relations = pd.DataFrame(np.sort(relations.values, axis=1), columns=relations.columns)

    # calculating the weights for the edges
    relations['value'] = 1
    relations = relations.groupby(["source", "target"], sort=False, as_index=False).sum()

    # creating a graph G
    G = nx.from_pandas_edgelist(relations,
                                source="source",
                                target="target",
                                edge_attr="value",
                                create_using=nx.Graph())

    # displaying the graph
    net = Network(notebook=True, width="1000px", height="700px", bgcolor='#222222', font_color='white')

    node_degree = dict(G.degree)
    communities = cl.best_partition(G)
    nx.set_node_attributes(G, communities, 'group')
    nx.set_node_attributes(G, node_degree, 'size')
    net.from_nx(G)

 
    
    html= net.generate_html()
    # Load HTML file in HTML component for display on Streamlit page
    components.html(html, width=900, height=700)
