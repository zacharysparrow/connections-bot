from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import spectral_embedding
import numpy as np
#from sklearn.metrics.pairwise import linear_kernel
#from sklearn.metrics.pairwise import rbf_kernel
import csv
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
#import fasttext.util
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.manifold import MDS
from cluster_equal_size import cluster_equal_size_mincostmaxflow
#from freedictionaryapi.clients.sync_client import DictionaryApiClient
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.cluster import SpectralClustering

large_width = 400
np.set_printoptions(linewidth=large_width)

def get_definition(word):
    defns = []
    with DictionaryApiClient() as client:
        parser = client.fetch_parser(word)
    word = parser.word
    for meaning in word.meanings:
        for definition in meaning.definitions:
                defns.append(definition.definition)
    return(defns)

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

puzzle = read_csv("puzzles/test_puzzle_3.txt")
puzzle_words = [x.lower().strip() for xs in puzzle for x in xs]
print(puzzle_words)

#fasttext.util.download_model('en', if_exists='ignore')  
#ft = fasttext.load_model('cc.en.300.bin')
#
#vecs = [ft.get_word_vector(w) for w in puzzle_words]

#sim_mat = cosine_similarity(vecs)
#sim_mat = linear_kernel(vecs)
#sim_mat = rbf_kernel(vecs, gamma=1/10)
model = SentenceTransformer("all-MiniLM-L6-v2")
#model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
#model = CrossEncoder("cross-encoder/stsb-TinyBERT-L4")

synsets = [wn.synsets(w) for w  in puzzle_words]
#definitions = [get_definition(puzzle_words[i]) + [w.definition() for w in s] for i,s in enumerate(synsets)]
definitions = [[w.definition() for w in s] for i,s in enumerate(synsets)]
#definitions = [[] for i,s in enumerate(synsets)]
#definitions = [get_definition(w) for w in puzzle_words]
for i,d in enumerate(definitions):
    d.append(puzzle_words[i])

#print()
#for s in synsets:
#    print(s)
#
#sim_mat = [[0 for j in range(16)] for i in range(16)]
#for wi,w in enumerate(synsets):
#    for xi,x in enumerate(synsets):
#        all_sims = []
#        for ws in w:
#            for xs in x:
#                all_sims.append(wn.wup_similarity(ws,xs))
#        sim_mat[wi][xi] = max(all_sims)

dist_mat2 = [[0.0 for j in range(16)] for i in range(16)]
embeddings = [model.encode(defn) for defn in definitions]

print()
for d in definitions:
    print(d)

def find_max_position(matrix, idx):
    matrix_np = np.array(matrix)
    max_index_flat = np.argmax(matrix_np)
    row_index, col_index = np.unravel_index(max_index_flat, matrix_np.shape)
    return [int(row_index), int(col_index)][idx]

save_pair = [5,6]
save_sims = []
defns_used = [['x' for j in range(16)] for i in range(16)]
#defns_used2 = [[0 for j in range(16)] for i in range(16)]
chosen_defns = [0,1,0,2,2,2,1,4,3,7,5,2,4,2,0,1] #6, 12, 14, 15 are arbitrary
for wi,w in enumerate(embeddings):
    for xi,x in enumerate(embeddings):
        if xi <= wi:
            continue
        all_sims = [[0.0 for j in x] for i in w]
        for wj,ws in enumerate(w):
            for xj,xs in enumerate(x):
                all_sims[wj][xj] = cosine_similarity([ws],[xs])[0][0]
#                all_sims.append(cosine_similarity([ws],[xs])[0][0])
#        dist_mat2[wi][xi] = 1-max(all_sims) #uncomment to make dist matrix
#        print([wi,xi])
#        print(all_sims)
        if wi == save_pair[0] and xi == save_pair[1]:
            save_sims = all_sims
#        dist_mat2[wi][xi] = max([max(x) for x in all_sims]) #uncomment to make affinity matrix
        dist_mat2[wi][xi] = all_sims[chosen_defns[wi]][chosen_defns[xi]] #uncomment to make affinity matrix
#        defns_used[wi][xi] = find_max_position(all_sims,0) 
#        defns_used[xi][wi] = find_max_position(all_sims,1)
#        dist_mat2[wi][xi] = max(all_sims) #uncomment to make affinity matrix
#        dist_mat2[wi][xi] = np.mean(all_sims) #uncomment to make affinity matrix
#        dist_mat2[wi][xi] = np.mean([x**2 for x in all_sims]) #uncomment to make affinity matrix
        dist_mat2[xi][wi] = dist_mat2[wi][xi]

print("looking for consistency in number accross rows within (expected) cluster")
print(np.array(defns_used))

print(save_sims)
plt.hist(sum(save_sims,[]))
plt.show()

#for wi,w in enumerate(definitions):
#    for xi,x in enumerate(definitions):
#        if xi < wi:
#            continue
#        all_sims = []
#        for ws in w:
#            ranks = model.rank(ws, x)
#            scores = [rank['score'] for rank in ranks]
#            all_sims.append(max(scores))
##        for xs in x:
##            ranks = model.rank(xs, w)
##            scores = [rank['score'] for rank in ranks]
##            all_sims.append(max(scores))
##        all_sims = max(all_sims)
#        dist_mat2[wi][xi] = 1-max(all_sims)
#        dist_mat2[xi][wi] = dist_mat2[wi][xi]

for i,row in enumerate(dist_mat2):
    dist_mat2[i][i] = 1.0

#cluster_size = 4
#n_clusters = 4
#unassigned = range(cluster_size*n_clusters)
#clusters = [[] for n in n_clusters]
#for 

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
# Get the embeddings
X_transform = mds.fit_transform(dist_mat2)

eigenvecs = spectral_embedding(np.array(dist_mat2), n_components=4, random_state=0)
#print(eigenvecs)

#clustering = SpectralClustering(n_clusters=4,
#        assign_labels='discretize',
#        affinity='precomputed',
#        random_state=0).fit(dist_mat2)
#predicted_groups = clustering.labels_

#clustering = SpectralClustering(n_clusters=3,
#        assign_labels='discretize',
#        affinity='precomputed',
#        random_state=0).fit([x[0:12] for x in dist_mat2[0:12]])
#predicted_groups = clustering.labels_
#predicted_groups = cluster_equal_size_mincostmaxflow(X_transform, 4, show_plt=False)[0]#[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
predicted_groups = cluster_equal_size_mincostmaxflow(eigenvecs, 4, show_plt=False)[0]#[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
print(predicted_groups)

def group_by_cluster(points, labels):
    grouped_points = defaultdict(list)
    for i, label in enumerate(labels):
        grouped_points[label].append(points[i])
    return dict(grouped_points)

def average_distance_in_group(group):
    if len(group) < 2:
        return 0
    distances = [np.sqrt(sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))
                 for p1, p2 in combinations(group, 2)]
    return np.mean(distances)

grouped_data = group_by_cluster(eigenvecs, predicted_groups)
grouped_words = group_by_cluster(puzzle_words, predicted_groups)
print(grouped_words)
spreads = []
for g in list(grouped_data.values()):
    spreads.append(average_distance_in_group(g))
print(spreads)

#smallest_cluster = spreads.index(min(spreads))

##plt.matshow(sim_mat)
##plt.colorbar()
##plt.title("Word Similarities synsets")
##plt.show()
#sizes = [64 for i in range(16)]
#colors = ['y','y','y','y','g','g','g','g','b','b','b','b','m','m','m','m']
#group_colors = [['y','g','b','m'][i] for i in predicted_groups]
#fig = plt.figure()
##ax = fig.add_subplot(projection='3d')
#ax = fig.add_subplot()
##ax.scatter(X_transform[:,0], X_transform[:,1], X_transform[:,2], s=sizes, c=colors)
#ax.scatter(X_transform[:,0], X_transform[:,1], s=[4*x for x in sizes], c=group_colors)
#ax.scatter(X_transform[:,0], X_transform[:,1], s=[2*x for x in sizes], c=['k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k'])
#ax.scatter(X_transform[:,0], X_transform[:,1], s=sizes, c=colors)
#plt.title('Embedding')
#plt.show()

plt.matshow(dist_mat2)
plt.colorbar()
plt.title("Word Similarities defn Transformer")
plt.show()
