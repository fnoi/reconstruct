


with open(pkl_name, 'rb') as file:
    try:
        while True:
            juergen = pickle.load(file)
    except EOFError:
        pass