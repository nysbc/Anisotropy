### 3DFSC Program Suite Version 3.0 (16 March 2018) ###

This is the most recent version of 3DFSC, by Philip Baldwin, Yong Zi Tan, and Dmitry Lyumkis.
GPU code and Conda environment by Carl Negro.
3DFSC Program Suite requires Anaconda 3 to run, and UCSF Chimera to visualize the outputs.

## Installation ##
1) Be in the desired directory where you want to install the 3DFSC Program Suite.
2) Type `git clone https://github.com/nysbc/Anisotropy.git` to get a copy of the program. Make sure you have the git package installed in your Linux machine else this would not work.
3) Type `cd Anisotropy` to enter the directory.
4) Type `conda env create -f environment.yml` to create the Anaconda environment containing required packages. This would only work if Anaconda 3 is installed and used as the default Python. You do not need to activate the new environment yet.
5) Edit the file `run3DFSC.csh` with a text editor to reflect your 3DFSC directory (which contains `ThreeDFSC_Start.py`). You can copy this file anywhere to start the 3DFSC program. It might be a good idea to make an alias to this `run3DFSC.csh` file.

## Execution ##

1) Be in the directory containing your maps. Relative paths are okay for the program.
2) Execute the `run3DFSC.csh` script. If no options are given, it will print out the help menu for you.
3) Runs usually take from minutes up to hours for extremely large box sizes (we have tested 600^3). Progress bars will help indicate the state of processing.
4) 3DFSC is also accessible programmaticaly, for example:
    <pre>import Anisotropy.ThreeDFSC.ThreeDFSC_Start as run_fsc
    run_fsc.execute(options)</pre>
    
    where "options" is an opt-parser object containing the parameters of the program.

## Example: Haemagglutinin Trimer with Preferred Orientation Collected at Tilts ##

1) Go the Example directory
2) Type `../run3DFSC.csh --halfmap1=T40_map1_Masked_144.mrc --halfmap2=T40_map2_Masked_144.mrc --fullmap=130K-T40.mrc --apix=1.31 --ThreeDFSC=T40-3DFSC` to run the Example. It should take no longer than 2 minutes.
3) The folder already contains pre-calculated results as well as a log of a successful run (`output.log`).

Questions and feedback welcomed, and should be sent to prbprb2@gmail.com, ytan@nysbc.org and dlyumkis@salk.edu.
