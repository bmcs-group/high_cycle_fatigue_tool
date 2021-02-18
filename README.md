# High-Cycle-Fatigue-Tool (HCFT or HcFatFit)
A tool with GUI (Graphical User Interface) for processing CSV files obtained from fatigue experiments up to the high-cycle fatigue ranges. Additionally, tests with monotonic loading can be processed.

## Features:
1. Simple plot functionality for columns of the CSV file.
2. Extracting Max and Min values and filtering the undesired cycles.
3. Extracting and plotting the fatigue creep curve (cycles number vs displacement).
4. Smoothing function for fatigue creep curves.
5. Ability to process file with +20 Gb size.
6. Built-in tool for joining CSV (or TXT) files.
7. Graphical User Interface with all functions and parameters

## Usage:
<ul>
<li><b>Option 1 (Installing Python environment with all needed libraries)</b><br>
<ul>
<li>
Install Miniconda <a href="https://docs.conda.io/en/latest/miniconda.html">(download link)</a>.</li>

<li>Perform the following commands from Anaconda Command Prompt to install the required libraries:

`conda install -c anaconda matplotlib`

`conda install -c anaconda scipy`

`conda install -c anaconda pandas`

`conda install -c anaconda traits`

`conda install -c anaconda traitsui`

</li>

<li>Clone this repository or download its contents
</li>

<li>Run the tool using the command

`python PATH_TO_THE_TOOL_REPOSITORY/hcft.py`


</li>
</ul>

<li><b>Option 2 (Direct install of the tool from the exe installer file)</b>
<br>
You can download and install the tool on Windows using on of these installers:

Windows 64bit: <a href="https://github.com/ishomam/high-cycle-fatigue-tool/releases/download/v1.0/hcft_v1.0_64bit.exe">hcft_v1.0_64bit.exe
</a>

Windows 32bit: <a href="https://github.com/ishomam/high-cycle-fatigue-tool/releases/download/v1.0/hcft_v1.0_32bit.exe">hcft_v1.0_32bit.exe
</a>
</li>
</ul>


## Cite with: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3603816.svg)](https://doi.org/10.5281/zenodo.3603816)
The repository can refered to using a unique doi hosted at https://zenodo.org


## Screenshots:
![](screenshots/High_Cycle_Fatigue_Tool.png "HCFT tool")

![](screenshots/CSV_files_joiner.png "built-in CSV joiner tool")
