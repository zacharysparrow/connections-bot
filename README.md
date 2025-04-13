# connections-bot
Bot for solving the NYT Connections puzzles

Stack:
- python
- fasttext

Strategy:
- Look up the definition(s) of each word
- Embed each definition for each word
- If the word is not in the dictionary, just embed the word itself
- Compute similarities between each definition for each word
- Use definitions with maximum similarity between words for word similarities (can improve -- words only appear in one contex, so the definition used can be a combinatorial variable in clustering)
- Cluster words into groups of 4 based on similarity scores
