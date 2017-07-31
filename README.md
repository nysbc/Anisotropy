This is the most recent version of 3DFSC, by Philip Baldwin, Yong Zi Tan, and Dmitry Lyumkis.
Conda environment configured by Carl Negro.

Installation: "conda env create -f environment.yml"

3DFSC is compatible and tested with Python 3.5+

Command line execute of the script is possible with the command "./run_fsc" from the top-most directory (tested in both bash and csh).

3DFSC is also accessible programmaticaly, for example:

import ThreeDFSC.ThreeDFSC_Start as run_fsc

run_fsc.execute(options)

...where "options" is an opt-parser object containing the parameters of the program. Thanks,

--CJN

