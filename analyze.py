import csv
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

stats = []
with open('puzzles/solver_data.csv', 'r', newline='') as file:
    solver_data = csv.reader(file)
    for row in solver_data:
        stats.append(row)

guess_counts = [ast.literal_eval(s[0]) for s in stats]
found_colors = [ast.literal_eval(s[1]) for s in stats]
found_connections = [ast.literal_eval(s[2]) for s in stats]

n_puzzles = len(guess_counts)
n_solved = 0
for g in guess_counts:
    if g[0] == 4:
        n_solved += 1

solved_hist = []
for g in guess_counts:
    solved_hist.append(g[0])

color_counts = [0,0,0,0]
for c in found_colors:
    color_counts = [sum(x) for x in zip(color_counts, c)]

print(n_solved)
print(n_puzzles)
print(color_counts)

mean_depth = np.mean(solved_hist)

plt.figure(figsize=(10, 6))
hist = sns.histplot(data=solved_hist, bins=5, discrete=True)
y_bottom, y_top = hist.get_ylim()
padding = (y_top - y_bottom) * 0.1
hist.set_ylim(y_bottom, y_top + padding)
text_place = np.mean(sorted(hist.get_yticks())[0:-1])

plt.axvline(mean_depth, 0, 1, color='black')
hist.annotate("mean: "+'{0:.2f}'.format(mean_depth), xy=(mean_depth + 0.1,text_place), horizontalalignment = 'center', rotation=270)
for p in hist.patches:
    number_label = int(p.get_height())/len(solved_hist)
    hist.annotate(
        f'{number_label:.1%}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='center',
        xytext=(0, 10),
        textcoords='offset points'
    )

plt.title('Found Connections')
plt.xlabel('# Found')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

