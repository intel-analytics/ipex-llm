import faiss
import numpy as np

vectors = np.load("data.npy")

d = 128
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(vectors)                  # add vectors to the index
print(index.ntotal)
faiss.write_index(index, "flatl2.idx")

# index = faiss.read_index("flatl2.idx")
# D, I = index.search(vectors[:1], k=200)
