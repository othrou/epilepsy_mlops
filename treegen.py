import os

def generate_tree(startpath, prefix=''):
    for item in os.listdir(startpath):
        path = os.path.join(startpath, item)
        print(prefix + "|-- " + item)
        if os.path.isdir(path):
            generate_tree(path, prefix + "|   ")

generate_tree(".")
