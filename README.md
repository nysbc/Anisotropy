This is the most recent version of 3DFSC, by Philip Baldwin, Yong Zi Tan, and Dmitry Lyumkis.
Conda environment configured by Carl Negro.
3DFSC Program Suite requires Anaconda 3 to run, and UCSF Chimera to visualize the outputs.

## Installation ##
1) Be in the desired directory where you want to install the 3DFSC Program Suite
2) Type `git clone https://github.com/nysbc/Anisotropy.git` to get a copy of the program. Make sure you have the git package installed in your Linux machine.
3) Type "cd Anisotropy" to enter the directory.
4) Type "conda env create -f environment.yml" to create the Anaconda environment containing required packages. This would only work if Anaconda 3 is installed and used as the default Python
5) Edit the file run3DFSC.csh to reflect your Anaconda3 and 3DFSC directory (which contains ThreeDFSC_Start.py).

## Execution ##

1) Be in the directory containing your maps. Relative paths are okay for the program too.
2) Execute the run3DFSC.csh script. If no options are given, it will print out the help menu for you.
3) Runs usually take from minutes up to hours for extremely large box sizes (we have tested 600^3). Progress bars will help indicate the state of processing.

4) 3DFSC is also accessible programmaticaly, for example:
    import ThreeDFSC.ThreeDFSC_Start as run_fsc
    run_fsc.execute(options)
    
    where "options" is an opt-parser object containing the parameters of the program.

Questions and feedback welcomed, and should be sent to prbprb2@gmail.com, ytan@nysbc.org and dlyumkis@salk.edu
