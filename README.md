# connections-bot
Bot for solving the NYT Connections puzzles

Stack:
- python


** Most room for improvement is in choosing word definitions
- Brute force works, but is too slow. Can speed up by
	- faster code
	- filtering out definitions that have low similarity scores
	- Identify pairs of points that are too far apart to be connected and don't iterate through those
- Dictionary clearly missing some relevant definitions, mostly informal ones. Use wiktionary API?
	- https://en.wiktionary.org/api/rest_v1/#/Page%20content/get_page_definition__term_
- Just use contextual encoding on the list of words?

LINKS:
https://medium.com/@prescod/using-ai-to-solve-connections-59d42e47985b
https://arxiv.org/pdf/2411.05778v2
