from sklearn.metrics.pairwise import cosine_similarity
import csv
import matplotlib.pyplot as plt
import fasttext.util

def read_csv(file_path):
    data_array = []
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data_array.append(row)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return []
    except Exception as e:
         print(f"An error occurred: {e}")
         return []
    return data_array

puzzle = read_csv("puzzles/test_puzzle_1.txt")
puzzle_words = [x.lower().strip() for xs in puzzle for x in xs]
print(puzzle_words)

fasttext.util.download_model('en', if_exists='ignore')  
ft = fasttext.load_model('cc.en.300.bin')

vecs = [ft.get_word_vector(w) for w in puzzle_words]

sim_mat = cosine_similarity(vecs)

plt.matshow(sim_mat)
plt.colorbar()
plt.title("Word Similarities")
plt.show()
