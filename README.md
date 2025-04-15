# connections-bot
Bot for solving the NYT Connections puzzles

Stack:
- python
- fasttext

Strategy:
- (x) Look up the definition(s) of each word
- (x) Embed each definition for each word
- (x) Also just embed each word itself as it's own definition
- (x) Compute similarities between each definition for each word
- (x) Use definitions with maximum similarity between words for word similarities
	- want to choose only one definition for each word, possibly cluster definitions first
	- choose defn that makes the tightest clusters with as much separation as possible between clusters
- Cluster words into groups of 4 based on similarity scores

Clustering seems to be the most room for improvement
Best strategy is likely NOT finding the best clusters -- start with clusters that are more certain, and work from there.
- Need each cluster to have 4 members
- Need to be able to update clusters given additional info (e.g. "one away..." or not)
- Update clusters when a guess is successful
- Likely want more greedy clustering -- if points are close they are very likely to be related, we don't want purple guesses messing up yellow

Dependent on the relevent definitions actually being in our dictionary... not good for informal uses

fine-tune LLM using semi-supervised clustering/metric learning techniques

post-processing after clustering to form most likely guess?

Data TODO:
- scrape puzzles
- format puzzle database
