# connections-bot
Bot for solving the NYT Connections puzzles

Stack:
- python


** Most room for improvement is in choosing word definitions
- Brute force works, but is too slow. Can speed up by
	- faster code
	- filtering out definitions that have low similarity scores
	- Identify pairs of points that are too far apart to be connected and don't iterate through those
- Dictionary clearly missing some relevant definitions, mostly informal ones. A way to build in Wiki lookups, expand dictionary?
	- https://en.wiktionary.org/api/rest_v1/#/Page%20content/get_page_definition__term_
- Want some heuristic to find most likely definitions to be used
- Example sentence + contextual word embedding
- Search for sentences in corpus containing word, contextual embedding of word for each example, QR factorization or clustering to thin results
	- https://en.wiktionary.org/wiki/ghost -- see quotations


TODO:
- scrape puzzles
- format puzzle database

