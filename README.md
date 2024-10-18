# Multi-Compliance Mean Field Game with FBSDEs

These are codes and results for the research conducted with Prof Campbell in the Columbia Stats Department starting in May 2024. 

This repository is organized in the following way, reflecting the baby steps we took from the scratch to set up the 2-Period-2-Agent (2P2A) framework.

>- 1Period (1P1A -> 1P1A)
>- 2Period (2P2A x1 + 1P2A x2)
>- FinalReports (Overview + Details)

The first 2 folders include __codes__ of different model and/or parameter settings, a __Results__ folder containing the pdf/html visualized outputs, a __Best Models Saved__ folder containing the saved models/parameters in the forms of `.pt` files (for the sake of reproducibility), and maybe a __Wrong Results__ folder serving as back-ups. The last main folder includes the big-picture overview as well as a detailed version for math and algorithm. 

Universally, there is a `README` file under each main folder, detailing its contents and code instructions. Read them carefully before running codes or reviewing the results for each individual model. 

:warning: __Reminder:__
> Before running the code, please ensure:
>- the environment where you wanna run the codes has been properly activated;
>- you are under the correct file path;
>- the required packages have been installed/upgraded through following command 
>   ```pip install -r requirements.txt```.