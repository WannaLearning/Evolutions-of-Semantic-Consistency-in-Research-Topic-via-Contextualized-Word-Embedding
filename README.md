# Evolutions-of-Semantic-Consistency-in-Research-Topic-via-Contextualized-Word-Embedding
The code for the paper,  ***Evolutions of Semantic Consistency in Research Topic via Contextualized Word Embedding***.

# Code:
The files, ***DataCollectionAndPrepocess.py*** and ***ExtractCSPapers.py***, are used to process the MAG dataset, by which the FoS and papers in the computer science field is extracted.  
The file, ***GenerateBERTVec.py***, presents our method to create topic embeddings. The title, abstract, and FoS are concatenated as input to the BERT model, and the average of embeddings of the FoS is employed to represent the topic embeddings. Hence, for each topic, it is encoded by a greater number of embeddings, which can be taken as the semantic distribution of the topic.  
The file, ***CalculatedMetrics.py*** presents the functions utilized to compute self similarity (SSIM) and maximum explainable variance (MEV) , as well as the self distance (SDIS) employed in our paper.  
The file, ***ExperimentsAndResults.py***, generates our results.
