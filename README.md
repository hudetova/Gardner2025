# Chapter 2

## Overview

Reconstructing Gardner's (1987) Augmented Transition Network using modern knowledge graph tools and integrating LLMs in the neuro-symbolic reasoning pipeline. 

## Installation

```bash
pip install -r requirements.txt
```

## Usage

First, you need to create a neo4j graph database, with the data as specified in the `knowledge_graph.cipher` file.

Then, you need to specify environment variables for the analysis to run:

```bash
export NEO4J_URI=<your-neo4j-uri>
export NEO4J_USER=<neo4j-user>
export NEO4J_PASSWORD=<password>
export GEMINI_API_KEY=<your-gemini-api-key>
```

After that, you can run the programme:

```bash
python legal_analyzer.py
```

## Repo structure

`analysis.ipynb`: (WIP) jupyter notebook to analyze quantitatively analyze the results of the legal analyser and do error analysis.

`legal_analyzer.py`: the main class to run the programme.

`utils.py`: utility functions for the programme.

`knowledge_graph.cipher`: the cipher file to create the knowledge graph in neo4j.

`visualizations.dot`: the dot file to create the visualizations.

`logs/`: the logs of the analysis. For the baseline LLM, the logs are available in the `logs/naive_llm` folder. For the constrained LLM, the logs are available in the `logs/guided_llm` folder. `assets/`: the assets for the paper.

## Results

The logs of the analysis reported in the paper are available in the `logs` folder.
For the baseline LLM, the logs are available in the `logs/naive_llm` folder.
For the constrained LLM, the logs are available in the `logs/guided_llm` folder.
