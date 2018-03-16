#!/bin/sh

###A script to auto-generate run3DFSC executable files

echo "#!/bin/bash\n\n\
### An auto-generated shell script to run 3DFSC using Anaconda3 without altering the user's Python environment \n\n\
source activate fsc\n\n\
python $PWD/ThreeDFSC/ThreeDFSC_Start.py "$@" ### Change to your 3DFSC directory\n\n\
source deactivate" > run3DFSC.bash

echo "#!/bin/sh\n\n\
### An auto-generated shell script to run 3DFSC using Anaconda3 without altering the user's Python environment \n\n\
source activate fsc\n\n\
python $PWD/ThreeDFSC/ThreeDFSC_Start.py "$@" ### Change to your 3DFSC directory\n\n\
source deactivate" > run3DFSC.sh

echo "### An auto-generated csh script to run 3DFSC using Anaconda3 without altering the user's Python environment \n\n\
source activate fsc\n\n\
python $PWD/ThreeDFSC/ThreeDFSC_Start.py "$@" ### Change to your 3DFSC directory\n\n\
source deactivate" > run3DFSC.csh

chmod u+x run3DFSC.bash
chmod u+x run3DFSC.sh
chmod u+x run3DFSC.csh

