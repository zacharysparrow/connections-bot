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
	- choose defn that makes the tightest clusters with as much separation as possible between clusters (modularity or silhouette)
- (x) Cluster words into groups of 4 based on similarity scores


** Most room for improvement is in choosing word definitions
- can use information like 3 away to refine definitions after wrong guess? Something like, "if we got the last guess wrong, we should reconsider how we interpret some words"
- can brute force all definitions when down to last 2 connections?
- need a refined definition picking algorithm, current one is greedy (can try simulated annealing if we want to get something working, regardless of speed?).
- can build in definition picking into the MILP by introducing pairwise do-not-group constraints between definitions, careful to remove all definitions once a word is removed from the pool
- Dictionary clearly missing some relevant definitions, mostly informal ones. A way to build in Wiki lookups?

** Other improvements
- fine-tune LLM using semi-supervised clustering/metric learning techniques


TODO:
- scrape puzzles
- format puzzle database
