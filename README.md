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
- (x) Cluster words into groups of 4 based on similarity scores


** Most room for improvement is in choosing word definitions
- Brute force works, but is too slow. Can speed up by
	- clustering definitions
	- faster code
	- filtering out definitions that have low similarity scores
	- Explicitly exclude connections that contain two definitions from the same word
	- Identify pairs of points that are too far apart to be connected and don't iterate through those
	- Greedy Approach (as described before): This remains a very efficient option. Start with the closest pair, then iteratively add the point that best reduces the average distance of the growing set. While not guaranteed to be optimal, it's fast.
	- Cluster everything with fixed cluster size = 4, then pick the best of the lot that satisfies constraints?
- Dictionary clearly missing some relevant definitions, mostly informal ones. A way to build in Wiki lookups?

** Other improvements
- fine-tune LLM using semi-supervised clustering/metric learning techniques


TODO:
- scrape puzzles
- format puzzle database

get sim mat
compute all group combos
sort by score
find best one
filter groups by constraints
repeat
