import numpy as np
import ast
import h5py


def store_table(filename):
    table = dict()
    table['test'] = list(np.zeros(7,dtype=int))

    with h5py.File(filename, "w") as file:
        file.create_dataset('dataset_1', data=str(table))


def load_table(filename):
    file = h5py.File(filename, "r")
    data = file.get('dataset_1')[...].tolist()
    file.close()
    return ast.literal_eval(data)

filename = "embeddings_out.h5"
store_table(filename)
data = load_table(filename)
print(data)
