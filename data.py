import pickle

with open("faiss_index/index.pkl", "rb") as f:
    data = pickle.load(f)

print(data)

