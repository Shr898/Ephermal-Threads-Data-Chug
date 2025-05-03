import os
import json
import spacy
import logging
import requests  
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import community.community_louvain as community
from flask import Flask, request, send_file, jsonify, abort
from flask_cors import CORS


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


# Create a Flask app object
app = Flask(__name__)
CORS(app)


# Directory of <user> will store all the JSON and .html
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user")


# Load spaCy model
nlp = spacy.load("en_core_web_md")


# Flask app endpoint </analyze>
# It serves the url to graph and json
@app.route('/analyze', methods=['POST'])
def analyze_text():
    uploaded_file = request.files.get('file')
    if not uploaded_file or not uploaded_file.filename.endswith('.txt'):
        logging.warning("Invalid or missing file in request.")
        return jsonify({"error": "Please upload a valid '.txt' file"}), 400

    logging.info(f"Received file: {uploaded_file.filename}")

    text_content = uploaded_file.read().decode("utf-8")
    logging.info("Text file read and decoded.")

    processed_text = nlp(text_content)
    logging.info("Text processed with spaCy model.")

    # Extract human names from each sentence
    sentence_entity = []
    for sentence in processed_text.sents:
        person_names = [entity.text for entity in sentence.ents if entity.label_ == "PERSON"]
        sentence_entity.append({
            "sentence": sentence.text.strip(),
            "characters": person_names
        })

    logging.info(f"Extracted {len(sentence_entity)} sentences with named entities.")

    # Create a path object.
    # The json files path will be saved here.
    json_filename = os.path.join(BASE_DIR, "sentence_entities.json")
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(sentence_entity, json_file, indent=2)

    logging.info(f"Saved named entities to {json_filename}")

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

    logging.info(f"Extracted {len(character_relations)} character relationships.")

    relations_dataframe = pd.DataFrame(character_relations)
    sorted_relations = pd.DataFrame(
        np.sort(relations_dataframe.values, axis=1),
        columns=["source", "target"]
    )


    # Create a weight dataframe that stores weight as a column.
    # Performs a Group By Sum() aggregation to count the occurences of a { "source": "target" } Group.
    sorted_relations['weight'] = 1
    grouped_relations = sorted_relations.groupby(["source", "target"], sort=False, as_index=False).sum()


    # Create a path object.
    # The weights will be stored in this <weights.json> file.
    weights_filename = os.path.join(BASE_DIR, "weights.json")
    with open(weights_filename, "w", encoding="utf-8") as file:
        json.dump(grouped_relations.to_dict(orient="records"), file, indent=2)

    logging.info(f"Saved named weights to {weights_filename}")


    # Create a { "source": "target" } undirected graph object.
    character_graph = nx.from_pandas_edgelist(
        grouped_relations,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.Graph()
    )

    node_sizes = dict(character_graph.degree)
    communities = community.best_partition(character_graph)
    nx.set_node_attributes(character_graph, node_sizes, 'size')
    nx.set_node_attributes(character_graph, communities, 'group')


    # Create a path object.
    # Store the degrees into <degrees.json> file.
    degrees_filename= os.path.join(BASE_DIR, "degrees.json")
    with open(degrees_filename, "w", encoding="utf-8") as file:
        json.dump(node_sizes, file, indent=2)

    logging.info(f"SAVED DEGREES JSON TO {degrees_filename}")
    logging.info(f"Graph generated with {len(character_graph.nodes)} nodes and {len(character_graph.edges)} edges.")

    network_visualization = Network(
        notebook=False,
        width="1000px",
        height="700px",
        bgcolor="#222222",
        font_color="white"
    )
    network_visualization.from_nx(character_graph)


    # Create a path object.
    # It stores the path to the html file.
    html_filename = os.path.join(BASE_DIR, "network.html")
    network_visualization.save_graph(html_filename)

    logging.info(f"Saved network graph to {html_filename}")


    # Return the url of json and graph.
    # This will return the html file as well as the json file.
    return jsonify({
        "json_file": "/download/json",
        "graph_file": "/download/graph"
    })


# JSON
# This endpoint serves json file.
# It will return the json file as it is.
@app.route('/download/json', methods=['GET'])
def download_json():
    return send_file(os.path.join(BASE_DIR, "sentence_entities.json"), mimetype = "application/json")


# WEIGHTS
# This endpoint will serve the relationship weights.
# It will return a json with integer weights and 2 names constituting a relationship.
@app.route('/download/relation_weights', methods=['GET'])
def relation_weights():
    weightsJSON= os.path.join(BASE_DIR, "weights.json")

    if os.path.exists(weightsJSON):
        return send_file(weightsJSON, mimetype= "application/json")

    return jsonify({"error": "weights.json not found"}), 404


# DEGREE
# This endpoint will serve the degree of the individual nodes/entities.
# A degree is a measure of well-connected a node is in a graph.
@app.route('/download/degree', methods= ['GET'])
def degree_Node():
    degreesJSON= os.path.join(BASE_DIR, "degrees.json")

    if os.path.exists(degreesJSON):
        return send_file(degreesJSON, mimetype= "application/json")

    return jsonify({"error": "degrees.json not found"}), 404


# NETWORK
# This endpoint serves network/graph file.
# It will return the .html file as it is
@app.route('/download/graph', methods=['GET'])
def download_graph():
    html_path = os.path.join(BASE_DIR, "network.html")

    if not os.path.exists(html_path):
        return jsonify({"error": "network.html not found"}), 404

    with open(html_path, "r", encoding= "utf-8") as f:
        html= f.read()

    custom_script= """
        <script type="text/javascript">
        setTimeout(function () {
            if (typeof network !== "undefined") {
            network.on("click", function (params) {
                if (params.nodes.length > 0) {
                const nodeName = params.nodes[0];
                window.parent.postMessage({ type: "NODE_CLICKED", nodeName }, "*");
                }
            });
            }
        }, 1000);
        </script>
        </body>
    """

    html= html.replace("</body>", custom_script)

    with open(html_path, "w", encoding= "utf-8") as file:
        file.write(html)

    return send_file(os.path.join(BASE_DIR, "network.html"), mimetype = "text/html")


# SUMMARY
# This endpoint will request summary flask app.
# It will return a summary json
@app.route("/download/summary", methods=['POST'])
def getSummary():
    request_data= request.get_json()
    name= request_data.get("name")
    if not name:
        return jsonify({"error": "Missing a name in request."}), 400

    json_path = os.path.join(BASE_DIR, "sentence_entities.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "Sentences json doesnt exist."}), 404

    with open(json_path, 'r') as file:
        data= json.load(file)

    sentences= [entry["sentence"] for entry in data if any(name.lower() in n.lower() for n in entry["characters"])]

    if not sentences:
        return jsonify({"error": f"No sentences found for name '{name}'"}), 404

    files = {'file': ('summary.json', json.dumps(sentences), 'application/json')}
    try:
        response = requests.post("http://host.docker.internal:5000/summarize", files=files)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logging.error(f"Failed to connect to summary service: {e}")
        return jsonify({"error": "Failed to summarize text."}), 500

# One of the most important code block.
# It seves to debug and also run specific commands.
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')