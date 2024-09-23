# 2Period (Folder Structure And Decription)
- ## `Joint_Optim_2Prdx1`
    > The agents would optimize their production for 2 (multiple) periods as a whole, planning ahead for the future periods. This displaying how such a long-term perspective would impact both their own strategies and the market equilibrium price. 
    >
    > It is organized as follows. 
    - ### `.py`
        > The `.py` files are the self-defined modules warpping up essential classes (models, parameters, etc.) and utillity functions, for the sake of simplicity and convenience.  
        >- __Model.py__
        > NN model and paramters are structured as classes, which can then be repeatedly called. See details in the later sections: `Main_Models()`, `Network()`, and `Params()`. 
        >- __utils.py__
        > Useful functions like getting forward losses, process trajactories, as well as visualization by calling `plot()`.

    - ### `__pycache__`
        - #### `.pyc`
            > The pycache files for self-defined modules. 
            >- __Model.cpython-311.pyc__
            >- __utils.cpython-311.pyc__
            
    - ### `.ipynb` 
        > In the name structures of "Adamax\_{_tricks_}\_{_loss\_types_}.ipynb", where there are 4 combinations of different learning targets and loss functions. 
        >
        >- __Adamax_clamp_ind_MSE.ipynb__
        >- __Adamax_clamp_ind_BCE.ipynb__
        >- __Adamax_clamp_sig_MSE.ipynb__
        >- __Adamax_logit_BCElogit.ipynb__
        >
        > Since these codes have been well wrapped, we can run different models with different targets and loss combos by simply adjusting a few paramters, which will be detailed in the later sections: `Main_Models()`, `Network()`, `Params()`, and `plot()` etc. 

    - ### `Results`
        > The results for different models with different parameters. Includes: 
        - #### `Best Models Saved`
            >  Where NN models are saved for the sake of being easily loaded (no need to be run again). More specifically, it contains several sub-folders named identically as the `.pdf` files, under which there are 2 `.pt` files for the 2 populations (i.e. pop1 and pop2). 
            >
            > For instance:
            >- __sigmoid_ind_0.0001lr_500steps_BCE_0.5w__
            >   - __pop1.pt__
            >   - __pop2.pt__
            >  
            
        - #### `.pdf` / `.html`
            > The exported model outputs: plots of forward losses, inventory in stock overtime time, the decomposed inventory accumulating (rates), market prices overtime, and histograms displaying the terminal model convergence. 
            >
            > For instance:
            >-  __sigmoid_ind_0.0001lr_500steps_BCE_0.5w.pdf__
            > It means that the NN models for initial values $z_0, v_0, u_0, y_0$ are clapmed within interval $[0,1]$ using sigmoid function. And the learning target for the terminal values are indicator funcitons, with a learning rate of 0.0001. After stepping through 500 epochs, the BCElosses are then calculated, whose gradiants will be got through backproagation `loss.backward()`.        
            > 
    - ### Wrong Models and Bad Results / WrongResults
        > The back-ups for the failed models and wrong results during research (or simply the old versions were dumped here after every update). The cuases may be annotated behind the underscore ('_'). 
        >
        > The contents are organized in the same way as __Results__, except for __Best Models Saved__ being replaced with __Bad Models Saved__ here. 

- ## Separate_Optim_1Prdx2
    > The agents would treat each end of period as the "end of world", considering only for meeting the quota for the current period, having no plan for the future. This is carried out by running the _single-period_ strategy twice. 
    >
    > Such short-sighted perspectives serve as a comparison to the long-term ones above, showing the differences it'd make to their own production as well as the whole market. 
    >
    > This folder is organized in the same structure as __Joint_Optim_2Prdx1__.