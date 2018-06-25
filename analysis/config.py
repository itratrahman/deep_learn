def append_path(path):

    '''a function that adds the given directory to system path'''

    ##Importing the system or operating system library
    import sys
    import os

    ##Extracting the absulute path of the directory
    path = os.path.abspath(path)

    ##If the path is not in the system path
    #then add the path to system path
    if path not in sys.path:

        sys.path.append(path)
