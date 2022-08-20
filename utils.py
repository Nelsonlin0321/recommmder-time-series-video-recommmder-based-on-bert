import pickle
def save_object(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def open_object(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def to_device(inputs,device='cuda'):
    
    if isinstance(inputs,dict):
        inputs = {k:v.to(device) for (k,v) in inputs.items()}
    else:
        inputs = inputs.to(device)
        
    return inputs