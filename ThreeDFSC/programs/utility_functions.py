import os
import sys


def print_progress(iteration, total, prefix='', suffix='', decimals=1):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration    - Required    : current iteration (Int)
        total        - Required    : total iterations (Int)
        prefix        - Optional    : prefix string (Str)
        suffix        - Optional    : suffix string (Str)
        decimals    - Optional    : positive number of decimals in percent complete (Int)
        bar_length    - Optional    : character length of bar (Int)
    """

    #rows, columns = os.popen('stty size', 'r').read().split()
    columns = 40
    bar_length = int(float(columns)/2)
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total))) ## adjusted base on window size
    bar = '*' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\x1b[2K\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


## Added to mute the print statements
def blockPrint():
        sys.stdout = open(os.devnull, 'w')
        pass

def enablePrint():
#       sys.stdout = open('threedfscstdout.log', 'a')
        #sys.stdout = sys.__stdout__
        pass

