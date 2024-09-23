# `Model` and `utils` (Module Handbook)

- ##  `Params()`
    ```python
    CLASS Model.Params(param_type, target_type,trick,loss_type, delta,w=1.0,K=0.9,lr=0.005,NumTrain=500, T=2, NT1=50, NT2=100, device)
    ```

    Set global parameters for the model, which also facilitates the parameter tuning by simply calling and revising the attributes of `Params()`. 

    - __Parameters__: 
        - __param_type__ (_str_) - specifies the model parameter sets for an individual population. 
            - `'k1'`: the parameter set for sub-population 1. 
            - `'k2'`: the parameter set for sub-population 2.
            
            |       |$\pi_k$ | $h^k$ | $\sigma^k$ | $\zeta^k$ | $\gamma^k$ | $v^k$ | $\eta^k$ | $\beta^k$ |
            | :---: | :----: | :---: | :--------: | :-------: | :--------: | :---: | :------: | :--------:|
            |   k1  | 0.25   | 0.2   |  0.1       |   1.75    |   1.25     |  0.6  |  0.1     | 1.0       |
            |   k1  | 0.75   | 0.5   |  0.15      |   1.25    |   1.75     |  0.2  |  0.1     | 1.0       |

        - __target_type__ (_str_) - the target function for terminal values. Originally and mathematically, it should be the indicator functions. However, for the sake of numeric stability, we make use some tricks like sigmoid approximation and logit transformation, giving rise to the change of target and loss functions, accordingly.
            - `'indicator'`: learn the indicator functions (with jumps) directly. It would be the hardest to learn especially when `w`=1, which means the largest jump. Thus, models with indicator tagets would be least numercially stable. 
            - `'sigmoid'`: use $sigmoid(\frac{K-x}{\delta})$ to approximate $indicator(K>x)$, smoothing the jump from 0 to 1. The smaller the `delta` ($\delta$), the greater the 'slope' in the neighbourhood of $x=0$, the closer the approximation of the plain indicator function, yet the harder to learn. Note that there would be a trade-off of numeric stability and the difficulty of training.
            
        - __trick__ (_str_) - the trick used to increase the numeric stability and avoid exploding of gradients.
            - `'clamp'`: to avoid values falling out of the interval $[0,1]$ when learning indicator functions (or sigmoid approximations), we use `torch.Tensor.clamp` to forcefully clamp the values within $[0,1]$ (or $w*[0,1]$, more precisely). 
             - `'logit'`: a more numerically stable method to restrict values with $[0,1]$ by using $dY_t=(1-Y_t)Y_t Z_tdW_t$, transforming $\tilde{Y_t}=logit(Y_t)$, or $Y_t=sigmoid(\tilde{Y_t})$. And by $\text{It}\hat{\text{o}}$'s formula, $d\tilde{Y_t}=Z_t^2(Y_t-\frac{1}{2})dt+Z_t dW_t$. 

        - __loss_type__ (_str_) - the loss function paired with a specific trick and target function. 
            - `'MSELoss'`: Mean Squared Error loss. See more details in [`torch.nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#mseloss).
            - `'BCELoss'`: Binary Cross Entropy Loss between the target and the input probabilities (learned values). See more details in [`torch.nn.BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss).
            - `'BCEWithLogitsLoss'`: combines a Sigmoid layer and the BCELoss in one single class. See more details in [`torch.nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss).

        >__\* *NOTE*__: valid combiantions of `target_type`, `trick`, and `loss_type` are:
        >
         ```python
        {target_type: 'indicator', trick: 'logit', loss_type: 'BCEWithLogitsLoss'}  ## combo 1
        {target_type: 'indicator', trick: 'clamp', loss_type: 'BCELoss'}            ## combo 2
        {target_type: 'indicator', trick: 'clamp', loss_type: 'MSELoss'}            ## combo 3
        {target_type: 'sigmoid'  , trick: 'clamp', loss_type: 'MSELoss'}            ## combo 4
        ```

        - __delta__ (_float_) - $\delta$ of $sigmoid(\frac{K-x}{\delta})$, controlling the closeness of approximation, i.e. the smaller the $\delta$, the closer the approximation of indicator functions. Only valid when `target_type`='sigmoid'.

        - __w__ (_float, optional, defualt: 1.0_) - control of jump size. The real learning targets are $w*indicator(\cdot)$ or $w*sigmoid(\cdot)$, which would be easier to learn when $w<1.0$ since the jumps are narrowed from $[0,1]$ to $[0,w]$. Thus the smaller `w` is, the more numerically stable.

        - __K__ (_float, optinal, defualt: 0.9_) - the quota to meet at each end of period. And amount below the quato will be subjected to a penalty of $w*(K-X_T)$. Or more generally, the penalty is defined by a put option (ReLU) function: $w*(K-X_T)_+$. The choice of quato should be "_attainable_" - not too hard nor too easy to meet. 

        - __lr__ (_float, optinal, defualt: 0.005_) - learning rate. Should be adjusted to smaller values when targets are hard to learn, for the sake of better convergence.

        - __NumTrain__ (_int, optinal, defualt: 500_) - number of training samples within each sub-population. 

        - __T__ (_int/float, optinal, defualt: 2_) - the "end of world", or the terminal time for the last period, $T_2$. Since `T` represents a scaled time period, it'll influence the grid size $dt$ given a fixed number of time grids (`NT1` and `NT2`). Though $dt$ has no impact on any of the rate processes $g_t$, $\Gamma_t$, $a_t$, and price process $\S_t$, nevertheless, it will influence the accumulation of inventory, thus how hard it is to meet the quota `K`. Intuitively, given enough time, however small their generation rates are (however lazy they are), the agents will be bound to meet the quota. 

        - __NT1__ (_float, optinal, defualt: 50_) - the number of time grids for the first period.

        - __NT2__ (_float, optinal, defualt: 100_) - the number of time grids for the second period. The grid size is calculated as $dt=\frac{T_2}{{NT}_2}$, and the end of the first period is $T_1=dt*{NT}_1$
        
        - __device__ (_str_, optional) - the device to train NN models. It is defualtly decided by  device-agnostic code:
            ```python
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
            ```

- ## `Network()`
    ```python
    CLASS Network(scaler_type=None, input_dims=1, fc1_dims=10, fc2_dims=10, n_outputs=1)
    ```

    A subclass of [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). 
    
    Here we use fully connected linear layers for NN models, applying the affine linear transformations for incoming data:
    $$ 
    y=xA^T+b
    $$
    The weights of linear layers are initialized using [`torch.nn.init.xavier_uniform_()`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_). And the activate layers used here are [`torch.nn.ReLU()`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU).

    - __Parameters__:
        - __scaler_type__ (_str/NoneType, optinal, defualt: None_) - the method for scaling model outputs to $[0,1]$. Should only be used for models for initial values $V_0$, $U_0$, and $Y_0$. 
            - `None`: return the model output x without scaling.
            - `'minmax'`: return $\frac{x-x_{min}}{x_{max}-x_{min}}$ in the `forward()` function. Keep the original distribution of x.
            - `'sigmoid'`: return $sigmoid(x)$. Provide greater "differentiability" when computing gradients.
        - __input_dims__ (_int, optinal, defualt: 1_) - the size of each input sample (number of $x$ in the linear transformation). The same as `in_features` of [`torch.nn.Linear()`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear). 
        
        - __fc1_dims__ (_int, optinal, defualt: 10_) - the size of each output sample (number of $y$, or sets of weights $A^T$ and biases $b$) for the first fully-connected layer. The same as `out_features` of [`torch.nn.Linear()`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear). Also the number of inputs of the second layer. 

        - __fc2_dims__ (_int, optinal, defualt: 10_) - the number of outputs of the second fully-connected layer, also the number of inputs of the third layer.

        - __n_outputs__ (_int, optinal, defualt: 1_) - the number of outputs of the last layer.

    - > __forward__ ( _input_ )

        Piping the inputs through [`torch.nn.Sequential()`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential). And the outputs are scaled accordingly when `scaler_type` is specified. 
        ```python
        def forward(self,input):
        model=nn.Sequential(self.fc1,
                            self.relu1,
                            self.fc2,
                            self.relu2,
                            self.fc3).to(self.device)
        x=model(input)
        if self.scaler_type=='minmax':
            return ((x-x.amin())/(x.amax()-x.amin())).to(self.device)
        if self.scaler_type=='sigmoid':
            return torch.sigmoid(x).to(self.device)
        if self.scaler_type==None:
            return x.to(self.device)
        ```
- ## `Main_Models()`
    ```python
    CLASS Main_Models(GlobalParams)
    ```

    Define the NN models for each discretized time point. Models can be saved and/or loaded repeatedly. 

    - __Parameters__:
        - __GlobalParams__ (_Params_) - the parameters for a specific NN model defined by `Params()` instances. 
    - __Attributes__:
        - __GlobalParams__ (_Params_) - the parameters for a specific NN model defined by `Params()` instances. 
        - __loss__ (_list/NoneType_) - the average forward losses after each epoch. Useful when saving and loading the models. 
        - __dB__ (_tensor_) - the independent Brownian Motion increments generated by `utils.SampleBMIncr()`. Shape: (`Numtrain`, `NT2`+1). Should be the same within the same subpopulation yet vary across subpopulations.
        - __init_x__ (_tensor_) - the initial values of inventory $X_0$, generated by `utils.Sample_Init()`. $X_0^{(k)} \sim \mathcal{N}(v^k, \eta^k)$. 
        - __init_c__ (_tensor_) - the initial increase to the baseline generation rate $C_0$, generated by `utils.Sample_Init()`. $C_0^{(k)} \equiv 0$.
        
    - > __create__ ( _kwargs_ )
        ```python
        Main_Models.create(y0_model,yt1_model,zy_models,forward_loss=None,dB=None,init_x=None,init_c=None)
        ```
        Set attributes for a `Main_Models()` instance. 
        - __Parameters__:
            - __y0_model__ (_Network_) - NN model for $Y_0$.
            - __yt1_model__ (_Network_) - NN model for $Y_{T_1}$. When 
