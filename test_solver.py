import csv
from sentence_transformers import SentenceTransformer
from solve_connections import solver


def import_puzzles():
    pattern = "Connection"
    with open("puzzles/all_puzzles.txt", "r") as file:
        lines = file.readlines()
        all_puzzles = []
        puzzle = []
        for line in lines:
            if line[0] in set(["🟡","🟢","🔵","🟣"]):
                connection = [word.strip().lower() for word in line.split(':')[1].split(",")]
                puzzle.append(connection)
            else:
                if puzzle != []:
                    all_puzzles.append(puzzle)
                puzzle = []
    return all_puzzles

all_puzzles = import_puzzles()

n_embeddings_per_word = 3
model = SentenceTransformer("all-MiniLM-L6-v2")
solutions = []
for i in range(len(all_puzzles)):
    solution = solver(all_puzzles[i], n_embeddings_per_word, model)
    solutions.append(solution)

with open('puzzles/solver_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(solutions)

print()
for i,s in enumerate(solutions):
#    print(all_puzzles[i])
    print(s)

