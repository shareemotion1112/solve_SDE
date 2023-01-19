
from varname import nameof

verbose = True

def pp(txt):
    if verbose is True:
        print(f"{nameof(txt)} : {txt}")
