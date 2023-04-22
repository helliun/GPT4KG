# GPT4KG: Knowledge Graph Extraction with GPT-4
GPT4KG is a Python library for extracting knowledge graphs from text using GPT-4. The library uses OpenAI's GPT-4 language model to parse text and extract entities and relationships. The resulting knowledge graph is outputted as a JSON file, which can be used for downstream processing.

## Features
Extract entities and relationships from text using GPT-4
Build a knowledge graph from extracted entities and relationships
Search the knowledge graph for related entities based on a query
Display the knowledge graph as a PNG image
Requirements
Python 3.7 or higher
PyTorch
OpenAI API key
SentenceTransformer
pydot
numpy
sklearn
Installation
To install GPT4KG, use pip:

```bash
pip install GPT4KG
```
Usage
Here is an example of how to use GPT4KG to extract a knowledge graph from text:

```python
import json
from GPT4KG import KnowledgeGraph
```

## Instantiate the KnowledgeGraph class with your OpenAI API key
kg = KnowledgeGraph(api_key="your_api_key")

## Text to extract the knowledge graph from
text = "You are an expert AI that extracts knowledge graphs from text and outputs JSON files with the extracted knowledge, and nothing more. Here's how the JSON is broken down..."

## Extract the knowledge graph from the text
kg.update_graph(text)

## Display the knowledge graph
kg.display_graph()

# Search the knowledge graph for related entities
results = kg.search("OpenAI")
print(results)
Contributing
We welcome contributions to GPT4KG! If you'd like to contribute to the library, please create a pull request on the Github repository.

License
GPT4KG is licensed under the MIT License. See LICENSE for more information.
