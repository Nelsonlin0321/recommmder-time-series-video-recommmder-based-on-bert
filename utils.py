import pickle
def save_object(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def open_object(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj