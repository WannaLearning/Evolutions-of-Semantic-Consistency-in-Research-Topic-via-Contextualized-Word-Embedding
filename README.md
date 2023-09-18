# Evolutions-of-Semantic-Consistency-in-Research-Topic-via-Contextualized-Word-Embedding
The code for the paper,  ***Evolutions of Semantic Consistency in Research Topic via Contextualized Word Embedding***.

# Code:
The files, ***DataCollectionAndPrepocess.py*** and ***ExtractCSPapers.py***, are used to process the MAG dataset, by which the FoS and papers in the computer science field are extracted.  The MAG dataset can be downloaded through ***https://www.aminer.cn/oag-2-1***.
  
The file, ***GenerateBERTVec.py***, presents our method to create topic embeddings.  
The title, abstract, and FoS are concatenated as input to the BERT model, and the average of embeddings of the FoS is employed to represent the topic embeddings. Hence, for each topic, it is encoded by a greater number of embeddings, which can be taken as a semantic distribution of the topic.  The BERT model is implemented by the PyTorch framework with the BERT-base-model.
  
The file, ***CalculateMetrics.py*** presents the code utilized to compute self similarity (SSIM) and maximum explainable variance (MEV) , as well as the self distance (SDIS) employed in our paper.  
The three metrics quantify the semantic consistency of topic embeddings of a topic. SSIM, MEV, and SDIS can be analyzed as the time series. The K-Means with dtw distance is employed to identify four general evolution pattern of semantic consistency, that is ***IM*** (Increase), ***DS*** (Decrese), ***U-shape*** (increse first then decrease), and ***Inverted U-shape*** (decrease first then increse). These patterns suggest the exploration and exploitation in a topic in semantic vector space.
  
The file, ***ExperimentsAndResults.py***, generates our results in Section 4 of our paper.
