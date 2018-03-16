#!/bin/sh

###A script to auto-generate run3DFSC executable files

echo "\n******************************************************************\n"
echo "Attempting to create anaconda environment....\n"
conda env create -f environment.yml

echo "Generating executable scripts...\n"

ENVNAME="3DFSC"
echo "#!/bin/bash\n\n\
### An auto-generated shell script to run 3DFSC using Anaconda3 without altering the user's Python environment \n\n\
source activate $ENVNAME\n\n\
python $PWD/ThreeDFSC/ThreeDFSC_Start.py "$@" ### Change to your 3DFSC directory\n\n\
source deactivate" > run3DFSC.bash

echo "#!/bin/sh\n\n\
### An auto-generated shell script to run 3DFSC using Anaconda3 without altering the user's Python environment \n\n\
source activate $ENVNAME\n\n\
python $PWD/ThreeDFSC/ThreeDFSC_Start.py "$@" ### Change to your 3DFSC directory\n\n\
source deactivate" > run3DFSC.sh

echo "### An auto-generated csh script to run 3DFSC using Anaconda3 without altering the user's Python environment \n\n\
source activate $ENVNAME\n\n\
python $PWD/ThreeDFSC/ThreeDFSC_Start.py "$@" ### Change to your 3DFSC directory\n\n\
source deactivate" > run3DFSC.csh

chmod u+x run3DFSC.bash
chmod u+x run3DFSC.sh
chmod u+x run3DFSC.csh

if [ -f "run3DFSC.bash" ] && [ -f "run3DFSC.sh" ] && [ -f "run3DFSC.csh" ]; then
  echo "3DFSC executable scripts have been generated for bash, csh, and sh."
  echo "Feel free to delete scripts that you will not use (e.g. csh and sh if you use bash)."
  echo "You can now move the run3DFSC.*sh executable script anywhere you like to run the program."
  echo "\n******************************************************************\n"
else
  echo "Error generating executable files. Please create an issue at https://github.com/nysbc/Anisotropy/issues."
  echo "\n******************************************************************\n"
fi
