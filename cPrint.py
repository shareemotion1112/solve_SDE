
from varname import nameof


def pp(txt, verbose = True):
    if verbose is True:
        print(f"{nameof(txt)} : {txt}")
