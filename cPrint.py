
from varname import nameof

verbose = False

def pp(txt):
    if verbose is True:
        print(f"{nameof(txt)} : {txt}")
