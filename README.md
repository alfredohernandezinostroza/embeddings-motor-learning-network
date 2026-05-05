# LLM Embeddings for the Motor Learning Citation Network

1. The file data/citation_network_selected.graphml corresponds to the result of motor-learning-network, but filtered so nodes with less than 5 edges are excluded.

2. The network is available online [here](https://alfredohernandezinostroza.github.io/citation-network-ncm/).

3. The topics were obtained by running `topic_modeling.py`. It's an implementation of BERTopic with Specter2.

4. A visualization of the discovered topics over the network is available [here](https://alfredohernandezinostroza.github.io/topics-ncm/)

5. The script did not save the embeddings. The upgraded version of the script that does save the embeddings, among other things, is `topic_modeling_new.py`. Note that the result is not the online version

6. The main other feature of this script is that it will not recalculate the embeddings if the input graph is the same.