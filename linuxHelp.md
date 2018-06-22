# Linux Help

A small list of commands to help you using a linux system:

`source ~/.bashrc`
this tells the command line to use the users bashrc config file as the current standpoint
Careful it will reset after a restart

`which python`
Tells you which version of python is currently used

`$PATH`
The Variable where the Path is saved, the ones in front will be executed first

`export PATH="$HOME/anaconda3/bin/:$PATH"`
To append the bin folder to the Path Variable

`vim <filename>`
To open editor with file. Exit with ESC :wq (w=saving, q=quit)

`mkdir <dirname>` 
To create directory

`touch <.filename>`
To create any kind of file, even with a dot at the beginning

`rm -r <directory name>`
To remove a directory, without the -r parameter to delte files

`ls -a`
View hidden files in a directory

`chmod +x <filename>`
To make a file executable, like an batch file (.sh) for example

`htop`
To see currently running Processes and the cpu/gpu usage

`sh <sh file>`
To execute a sh file


###Conda
`conda create -n <envName> pip python=3.6`
Create conda environment with name

`source activate <envName>`
To switch into environment with name


