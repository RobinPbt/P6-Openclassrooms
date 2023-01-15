# P6-Openclassrooms

This repository has been created in the context of the 6th project of my Data Scientist training with Openclassrooms. 

The goal of this project was to use a dataset of images of products and their descriptions to evaluate the feasibility of an automatic classification model to find the relevant category of a product.
This model would allow to create a marketplace without human intervention for products categorization.

The main idea was to extract features from both images and descriptions, proceed to a dimensionality reduction (T-SNE) and finally a KMeans clustering (with K = number of possible categories).
This clustering is then compared to the actual categories with an ARI score to assess if a classification model seems feasible.

For features extraction part, I tried several approaches:
- Image recognition:
  - SIFT
  - CNN with transfer learning
- NLP:
  - Bag of words: CountVectorizer, Tf-idf
  - Word embeddings: Word2vec, SpaCy
  - Language models with transfer learning: USE, BERT
  
Resulting max ARI scores for both image recognition and NLP are 0.45 and 0.46: it seems feasible to create the classification model with a mix of both approaches and by optimizing models for our dataset.

You can also find the final presentation of the project in the file "P6 - Support de présentation - 2022 08 18" (French)
