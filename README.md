### 3DFSC Program Suite Version 3.0 (16 March 2018) ###

This is the most recent version of 3DFSC, by Philip Baldwin, Yong Zi Tan, and Dmitry Lyumkis.

GPU code and Conda environment by Carl Negro.

3DFSC Program Suite requires Anaconda 3 to run, and UCSF Chimera to visualize the outputs.

## Installation ##

1) Navigate to the directory where you want to install the 3DFSC Program Suite.
2) Type `git clone https://github.com/nysbc/Anisotropy.git` to clone a copy of the program from the Github repository. Make sure you have git installed properly or else this will fail.
3) Type `cd Anisotropy` to enter the directory.
4) Run the init script by executing the command `./init.sh`. This will create the necessary Anaconda3 environment, auto-detect your repository path, and generate run3DFSC executable scripts for bash, sh, and csh terminal shells. This will only work if Anaconda 3 is installed and Anaconda's Python interpreter is on your PATH variable.


## Execution ##

1) Navigate to the directory containing your maps.
2) Copy the run3DFSC.bash, .csh or .sh script to this directory.
3) To view the 3DFSC parameters, access the help info like `./run3DFSC.sh -h`.
4) Execute the run3DFSC script with the appropriate parameters. It can take from minutes for small maps up to hours for extremely large box sizes (we have tested 600^3). Progress bars will help indicate the state of processing.

## GPU Execution ##

3DFSC now has GPU support through Numba for faster execution (typically ~10x faster than the CPU implementation). This requires that CUDA is installed correctly. See http://www.nvidia.com/Download/index.aspx. 

To make use of a GPU, simply append the `--gpu` flag as a parameter to your ./run3DFSC.sh script.

You can select which GPU to use for processing with the `--gpu_id` flag. E.g. `--gpu_id=2` to run 3DFSC on the GPU with index number 2. The program currently only allows a single GPU to be used at once; this may change in future versions.

To see a list of available GPU's and corresponding indices, run `nvidia-smi`. If you are unable to run `nvidia-smi`, check to make sure you have CUDA installed correctly. See http://www.nvidia.com/Download/index.aspx.

Note that GPU memory is limited, so that processing jobs with large box sizes will fail. 

## Example: Haemagglutinin Trimer with Preferred Orientation Collected at Tilts ##

1) Go the Example directory
2) Type `../run3DFSC.csh --halfmap1=T40_map1_Masked_144.mrc --halfmap2=T40_map2_Masked_144.mrc --fullmap=130K-T40.mrc --apix=1.31 --ThreeDFSC=T40-3DFSC` to run the Example. It should take no longer than 2 minutes (~15-20 seconds with a Pascal-generation GPU).
3) The folder already contains pre-calculated results as well as a log of a successful run (`output.log`).

Questions and feedback welcomed, and should be sent to prbprb2@gmail.com, ytan@nysbc.org and dlyumkis@salk.edu.
