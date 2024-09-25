# 2Period (Folder Structure And Decription)
- ## [Joint_Optim_2Prdx](Joint_Optim_2Prdx1)
    The agents would optimize their production for 2 (multiple) periods as a whole, planning ahead for the future periods. This displaying how such a long-term perspective would impact both their own strategies and the market equilibrium price. 
    &nbsp;
    It is organized as follows. 
    - ### [README.md](Joint_Optim_2Prdx1/README.md)
        Detailed user handbook and illustrations for its own `Model` and `utils` modules.
    - ### Illustration_diagrams
        Diagrams used in [README.md](Joint_Optim_2Prdx1/README.md) for the purpose of illustration.
    - ###  [\_\_pycache\_\_](Joint_Optim_2Prdx1/__pycache__)
        The pycache files for self-defined `Model` and `utils` modules. 
        - __Model.cpython-311.pyc__
        - __utils.cpython-311.pyc__
        - __Model.cpython-312.pyc__
        - __utils.cpython-312.pyc__

    - ### [Model.py](Joint_Optim_2Prdx1/Model.py) 
        NN model and paramters are structured as classes, which can then be repeatedly called. 
    - ### [utils.py](Joint_Optim_2Prdx1/utils.py)
        Useful functions like getting forward losses, process trajactories, as well as visualization by calling `plot()`.
        &nbsp;   
        > :bulb: __NOTE:__
        > The `*.py` files are the self-defined modules warpping up essential classes (models, parameters, etc.) and utillity functions, for the sake of simplicity and convenience. 
        > :bulb: See details in its own [README.md](Joint_Optim_2Prdx1/README.md). 

    - ### `*.ipynb` Files
        - [__Adamax_clamp_ind_MSE.ipynb__](Joint_Optim_2Prdx1/Adamax_clamp_ind_MSE.ipynb)
        - [__Adamax_clamp_ind_BCE.ipynb__](Joint_Optim_2Prdx1/Adamax_clamp_ind_BCE.ipynb)
        - [__Adamax_clamp_sig_MSE.ipynb__](Joint_Optim_2Prdx1/Adamax_clamp_sig_MSE.ipynb)
        - [__Adamax_logit_BCElogit.ipynb__](Joint_Optim_2Prdx1/Adamax_logit_BCElogit.ipynb)
        &nbsp;
        The file names follow the pattern of "Adamax\_{_tricks_}\_{_loss\_types_}.ipynb", corresponding to 4 valid combinations of learning targets, loss functions, and numeric tricks. 
        &nbsp;
        Well wrapped as they are, we can run different models and tune parameters easily, which will be detailed in [README.md](Joint_Optim_2Prdx1/README.md). 

    - ### [Results](Joint_Optim_2Prdx1/Results)  
        The results for different models with different parameters. Includes: 
        - #### [Best Models Saved](<Joint_Optim_2Prdx1/Results/Best Models Saved>)
            Where NN models are saved for the sake of being easily loaded (no need to be run again). More specifically, it contains several sub-folders named identically as the `.pdf` files, under which there are 2 `.pt` files for the 2 populations (i.e. pop1 and pop2). 
            
            :dizzy: __Illustration...__
            
            > - __sigmoid_ind_0.0001lr_500steps_BCE_0.5w__
            >   - __pop1.pt__
            >   - __pop2.pt__
            
        - #### `*.pdf` And `*.html` Files
            The exported model outputs: plots of forward losses, inventory in stock overtime time, the decomposed inventory accumulating (rates), market prices overtime, and histograms displaying the terminal model convergence. 
            
            :dizzy: __Illustration...__
            > - [__sigmoid_ind_0.0001lr_500steps_BCE_0.5w.pdf__](Joint_Optim_2Prdx1/Results/sigmoid_ind_0.0001lr_500steps_BCE_0.5w.pdf)
            > - [__sigmoid_sig_0.0005lr_0.03delta_500steps_MSE_0.5w.pdf__](Joint_Optim_2Prdx1\Results\sigmoid_sig_0.0005lr_0.03delta_500steps_MSE_0.5w.pdf)
            > - __...__

            > For the first example, it means that the NN models for initial values $z_0, v_0, u_0, y_0$ are clapmed within interval $[0,1]$ using sigmoid function. And the learning target for the terminal values are indicator funcitons, with a learning rate of 0.0001. After stepping through 500 epochs, the BCElosses are then calculated, whose gradiants will be got through backproagation `loss.backward()`.        
            
    - ### [Wrong Models and Bad Results](<Joint_Optim_2Prdx1/Wrong Models and Bad Results_no_dt>) / [WrongResults](Joint_Optim_2Prdx1/WrongResults_noClearance)
        The back-ups for the failed models and wrong results during research (or simply the old versions were dumped here after every update). The cuases may be annotated behind the underscore ('_'). 
        &nbsp;
        The contents are organized in the same way as [__Results__](Joint_Optim_2Prdx1/Results), except for [__Best Models Saved__](<Joint_Optim_2Prdx1/Results/Best Models Saved>) being replaced with [__Bad Models Saved__](<Joint_Optim_2Prdx1/WrongResults_noClearance/Bad Models Saved>) here. 

- ## [Separate_Optim_1Prdx2](Separate_Optim_1Prdx2)
    The agents would treat each end of period as the "end of world", considering only for meeting the quota for the current period, having no plan for the future. This is carried out by running the _single-period_ strategy twice. 
    &nbsp;
    Such short-sighted perspectives serve as a comparison to the long-term ones above, showing the differences it'd make to their own production as well as the whole market. 
    &nbsp;
    This folder is organized in the same structure as [__Joint_Optim_2Prdx1__](Joint_Optim_2Prdx1).
    &nbsp;
    :bulb: See details of module instructions in its own [README.md](Separate_Optim_1Prdx2/README.md). 