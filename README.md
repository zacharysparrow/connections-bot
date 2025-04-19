# connections-bot
Bot for solving the NYT Connections puzzles

Stack:
- python

Strategy:
- (x) Look up the definition(s) of each word
- (x) Embed each definition for each word
- (x) Also just embed each word itself as it's own definition
- (x) Compute similarities between each definition for each word
- (x) Use definitions with maximum similarity between words for word similarities
- (x) Cluster words into groups of 4 based on similarity scores


** Most room for improvement is in choosing word definitions
- Brute force works, but is too slow. Can speed up by
	- faster code
	- filtering out definitions that have low similarity scores
	- Identify pairs of points that are too far apart to be connected and don't iterate through those
- Dictionary clearly missing some relevant definitions, mostly informal ones. A way to build in Wiki lookups, expand dictionary?

** Other improvements
- fine-tune LLM using semi-supervised clustering/metric learning techniques


TODO:
- scrape puzzles
- format puzzle database

