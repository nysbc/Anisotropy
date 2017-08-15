### A shell script to run 3DFSC using Anaconda3 without altering the user's Python environment

source activate fsc

python ThreeDFSC/ThreeDFSC_Start.py "$@" ### Change to your 3DFSC directory

source deactivate
