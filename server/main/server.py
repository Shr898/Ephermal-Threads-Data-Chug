import os
import json
import spacy
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from flask import Flask, request, send_file, jsonify
import community.community_louvain as community


# Create a flask app object.
app = Flask(__name__)

# Create a nlp medium size model object.
nlp = spacy.load("en_core_web_md")  

# Create an endpoint the will recieve and trigger analysis code.
@app.route('/analyze', methods=['POST'])
def analyze_text():
    uploaded_file = request.files.get('file')
    if not uploaded_file or not uploaded_file.filename.endswith('.txt'):
        return jsonify({"error": "Please upload a valid '.txt' file"}), 400

    text_content = uploaded_file.read().decode("utf-8")
    processed_text = nlp(text_content)

    # Extract human names from each sentence
    sentence_entity = []
    for sentence in processed_text.sents:
        person_names = [entity.text for entity in sentence.ents if entity.label_ == "PERSON"]
        sentence_entity.append({
            "sentence": sentence.text.strip(),
            "characters": person_names
        })

    # Save sentence-entity JSON
    json_filename = "sentence_entities.json"
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(sentence_entity, json_file, indent=2)

    # Build character relationship graph
    sentence_dataframe = pd.DataFrame(sentence_entity)
    filtered_dataframe = sentence_dataframe[sentence_dataframe['characters'].apply(len) > 1]

    character_relations = []
    for index in range(filtered_dataframe.index[-1]):
        window_end = min(index + 5, filtered_dataframe.index[-1])
        characters_in_window = sum((filtered_dataframe.loc[index:window_end].characters), [])
        unique_characters = [characters_in_window[i] for i in range(len(characters_in_window))
                             if (i == 0) or characters_in_window[i] != characters_in_window[i - 1]]
        if len(unique_characters) > 1:
            for i in range(len(unique_characters) - 1):
                character_relations.append({
                    "source": unique_characters[i],
                    "target": unique_characters[i + 1]
                })

    relations_dataframe = pd.DataFrame(character_relations)
    sorted_relations = pd.DataFrame(
        np.sort(relations_dataframe.values, axis=1),
        columns=["source", "target"]
    )
    sorted_relations['weight'] = 1
    grouped_relations = sorted_relations.groupby(["source", "target"], sort=False, as_index=False).sum()

    # Create the network graph
    character_graph = nx.from_pandas_edgelist(
        grouped_relations,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.Graph()
    )

    # Add node size and community group
    node_sizes = dict(character_graph.degree)
    communities = community_louvain.best_partition(character_graph)
    nx.set_node_attributes(character_graph, node_sizes, 'size')
    nx.set_node_attributes(character_graph, communities, 'group')

    # Generate the HTML visualization using PyVis
    network_visualization = Network(
        notebook=False,
        width="1000px",
        height="700px",
        bgcolor="#222222",
        font_color="white"
    )
    network_visualization.from_nx(character_graph)
    html_filename = "network.html"
    network_visualization.save_graph(html_filename)

    return jsonify({
        "json_file": "/download/json",
        "graph_file": "/download/graph"
    })

@app.route('/download/json', methods=['GET'])
def download_json():
    return send_file("sentence_entities.json", as_attachment=True)

@app.route('/download/graph', methods=['GET'])
def download_graph():
    return send_file("network.html", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)