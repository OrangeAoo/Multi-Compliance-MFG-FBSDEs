# Multi-Period Compliance Mean Field Game with Deep FBSDE Solver
---

This is an research report giving big pictures about the problem we aim to solve, the key methods/algorithms we propose, the main results we get, as well as comparisons between different methods and consequent results. 

:bulb: See [_Report-StepwiseDetail_](../FinalReports/Report-StepwiseDetail.md) for more math and algorithm details; see _README_ files for more code instructions. 

---

## Abstract

The aim of this work is to extend the single-period compliance model in [[1]]("https://doi.org/10.48550/arXiv.2110.01127") to multi-period, proposing several tricks to improve the numeric stability of the deep solver for FBSDEs with jumps. First by reproducing the aformentioned research by Campbell, Steven, et al. (2021), then by considering an additional period to the original model, we make comparisons between long/short-term perspectives when it comes to multi-peirod production decision-making in renewable electricity certificate markets, as well as between different numeric tricks when it comes to algorithm stability. Meanwhile, some practical takeaways on parameter-tuning are recorded. 

## 1. Problem Overview

Conventional numerical solvers are hard pressed to solve PA-MFG with market-clearing conditions, which may be faced with the "curse of dimentionality" (Bellman 1957)[^1]. Thus in their study [[1]]("https://doi.org/10.48550/arXiv.2110.01127"), Professor Campbell and his fellows proposed an actor-critic approach to optimization, where the agents form a Nash equilibria according to the principal’s penalty function, and the principal evaluates the resulting equilibria. And they applies this approach to a stylized PA problem arising in Renewable Energy Certificate (REC) markets, where agents may _work_ overtime (or _rent_ capacity), _trade_ RECs, and _expand_ their long-term capacity to navigate the market at maximum profit. Here we only discuss the agents' problem in the multi-agent-multi-period scenario. 

### 1.1. REC Market Basics

Closely related to carbon cap-and-trade (C&T) markets, REC markets are a type of market-based emmissions regulation policies, which are motivating real-world applications of FBSDEs in modeling PA-MFG.

In RES markets, a regulator plays the role of principle, setting a floor on the amount of energy generated from renewable resources (aka. green energy) for each firm (based on a percentage of their total production), and providing certificates for each MWh of green energy generated and delivered to the grid. These certificates can be further traded by individual or companies, i.e. agents, to: 1) reduce costs or the greenhouse gas (GHG) emissions impact of their operations; and 2) earn profits from the extra inventories instead of wasting. Since the certificates are traded assets, energy suppliers can trade off between producing clean electricity themselves, and purchasing the certificates on the market. In all, such policies have played an important role in funding clean energy development, particularly in past years when the cost of green power production was not as competitive with the cost of fossil fuel power. 

To ensure compliance, each firm must surrender RECs totaling the floor at the end of a compliance period, with a monetary penalty paid for each lacking certificate. And in practice, these systems regulate multiple consecutive and disjoint compliance periods, which are linked together through a mechanism called _banking_, where unused allowances in current period can be carried on to the next period (or multiple future periods). Thus, as an extension to the single-period framework [[1]]("https://doi.org/10.48550/arXiv.2110.01127"), we now consider a 2-period model in this report.[^2]. 

### 1.2. REC Market Modeling with FBSDEs

Let's denote the 2 compliance periods $[0,T_1]$ and $(T_1,T_2]$ as $\mathfrak{T_1}$ and $\mathfrak{T_2}$, respectively. And $T_2$ can be thought of as 'the end of the world', after when there are no costs occurs and all agents forfeit any remaining RECs. Referring to steps in [[1]]("https://doi.org/10.48550/arXiv.2110.01127") and the probabilistic method in [[2]](https://arxiv.org/abs/1210.5780) (R. Carmona, F. Delarue, 2012) and considering the 2-agent-2-period MFG with market-clearing conditions, the optimal operation for agent $i$ in sub-population $k~(\forall~i \in \mathfrak{N}_k,~k\in\lbrace{1,2\rbrace})$ can be modeled with following coupled FBSDEs:

$$
\begin{cases}
    dX_t^{i} &=(h^{k}+g_t^{i}+\Gamma_t^{i}+C_t^{i})dt + \sigma^{k}dW_t^{k} - \min\left(X_{T_1}^i,K\right)\mathbf{1}_{t=T_1}&,  &X_0^{i} = \zeta^{i} \sim \mathcal{N}(v^k,\eta^k)\\
    dC_t^{i} &= a_t^{i}dt &,  &C_0^{k}=0 \\ 
    dV_t^{i} &= Z_t^{V,k}dW_t^{i}&,  &V_{T_1}^{i}=w*\mathbf{1}_{X^i_{T_1}<K} \\
    dU_t^{i} &= Z_t^{U,k}dW_t^{i}&,  &U_{T_1}^{i}=1*Y_{T_1}^i\mathbf{1}_{X^i_{T_1}>K}\\
    dY_t^{i} &= Z_t^{Y,k}dW_t^{i}&,  &Y_{T_2}^{i}=w*\mathbf{1}_{X^i_{T_2}<K}~~,
\end{cases} \\
$$

where the optimal controls are given by:

$$
\begin{aligned}
& g_t^{i} = \frac{V_t^{i}+U_t^{i}}{\zeta^{k}} ~\mathbf{1}_{t\in [0,T_1]}
            + \frac{Y_t^{i}}{\zeta^{k}} ~\mathbf{1}_{t\in (T_1,T_2]} \\
& \Gamma_t^{i} =\  \frac{V_t^{i}+U_t^{i}-S_t}{\gamma^{k}} ~\mathbf{1}_{t\in [0,T_1]}
                 + \frac{Y_t^{i}-S_t}{\gamma^{k}} ~\mathbf{1}_{t\in (T_1,T_2]} \\
& a_t^{i} =\frac{(T_1-t)(V_t^{i}+U_t^{i})+(T_2-T_1)Y^i_t}{\beta^{k}} ~\mathbf{1}_{t\in [0,T_1]}
            + \frac{(T_2-t)Y_t^{i}}{\beta^{k}} ~\mathbf{1}_{t\in (T_1,T_2]} \\
& S_t =\Biggl(
                \frac{\frac{\pi^1}{\gamma^1}}{\frac{\pi^1}{\gamma^1}+\frac{\pi^2}{\gamma^2}}\mathbb{E}[V_t^{i}+U_t^{i}|i \in \mathfrak{N}^1]+\frac{\frac{\pi^2}{\gamma^2}}{\frac{\pi^1}{\gamma^1}+\frac{\pi^2}{\gamma^2}}\mathbb{E}[V_t^{i}+U_t^{i}|i \in \mathfrak{N}^2]
            \Biggr) ~\mathbf{1}_{t\in [0,T_1]} + \Biggl(
                        \frac{\frac{\pi^1}{\gamma^1}}{\frac{\pi^1}{\gamma^1}+\frac{\pi^2}{\gamma^2}}\mathbb{E}[Y_t^{i}|i \in \mathfrak{N}^1]+\frac{\frac{\pi^2}{\gamma^2}}{\frac{\pi^1}{\gamma^1}+\frac{\pi^2}{\gamma^2}}\mathbb{E}[Y_t^{i}|i \in \mathfrak{N}^2]
                        \Biggr) ~\mathbf{1}_{t\in (T_1,T_2]} 
\end{aligned}
$$

The key notations/parameters are interpreted as follows: 

- $k \in \mathcal{K}$: a sub-population of agents, within which all individuals are assumed to have identical preferences and similar initial conditions/capacities, yet across which are distinct. The sub-population is annotated by superscript $[\cdot]^{k}$. Here we only discuss $k=1,2$.

- $i \in \mathfrak{N}$: an individual agent belonging to the sub-population $\mathfrak{N}^k$, annotated by superscript $[\cdot]^{i}$.

- $X_t := (X_t)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the current inventories in stock. For some key time points:
    - at $t=0$, there may be some stochastics in the initial inventories, which are assumed to be normally distributed. $X_0^{i} \sim \mathcal{N}(v^k, \eta^k) ,~ \forall k \in \mathcal{K},~\forall i \in \mathfrak{N}^k$.
    - at $t=T_1$, the terminal RECs pre-submission are $X_{T_1}$ carried over from the first period. Shortly after forfeiting $\min\Big(K,X^i_{T_1}\Big)$, the remaining inventories in stock are $ReLU\Big(X^i_{T_1}-K\Big)$, which are treated as new initial values for the second period.
    - at $t=T_2$, the terminal RECs pre-submission are $X^i_{T_2}$.

- $I_t := (I_t)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the integrated invetory generation. We introduce this process for continuous differentiablity at $T_1$. And $X_t$ has the same initial conditions as $I_t$. Clearly, we have:

    $$
    X_t=
    \begin{cases}
        & I_t~,                  ~~&& t \in [0,T_1]\\
        & I_t- \min(I_{T_1},K), ~~ && t \in (T_1,T_2]\\
    \end{cases} 
    ~~\text{or}~~ 
    X_t=
    \begin{cases}
        & I_t~,                                           ~~&& t \in [0,T_1]\\
        & I_t-I_{T_1}+(I_{T_1}-K)_+~, ~~&& t \in (T_1,T_2]\\
    \end{cases} 
    $$

- $K$: the quota that agents must meet at the end of each compliance period. Fixed to $K=0.9$[^3].

- $P(\cdot)$: the generic penalty function approximated by _**single-knot penalty functions**_[^4] : $$P(x)=w(0.9-x)_+ \Rightarrow\partial_{x}P(x) = - w\mathbf{1}_{x<K}.$$ Further, by tuning the weight $w$, we can see the relation between the penalty level (controled by $w$) and the agents' behaviour, as well as its market impact.

- $h$: the baseline generation rate at which agents generate with zero marginal cost. 

- $C_t := (C_t)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: incremental REC capacity of agents, i.e. the increase of baseline generation rate over time, accumulated by investing in expansion plans - for instance, by installing more solar panels. [^5]

- $a_t := (a_t)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the control of expansion rate, representing long-term REC capacity added per unit time. Note that it could be made even more realistic by incorporating a _delay_ between the decision to expand ($a_t$) and the increase to the baseline rate $h$.

- $g_t := (g_t)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the control of overtime-generation rate, i.e. the extra capacity achieved by working extra hours and/or renting short-term REC generation capacity at an assumed quadratic cost - specifically, overhour bonus and/or rental fees.

- $\Gamma_t := (\Gamma_t)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the control of trading rate, with negative[^6] values being the amount sold whereas postive purchased per unit time.

- $S_t := (S_t)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the equilibrium REC price obtained endogenounsly through market-clearing condition: 
$$\lim\limits_{N \to \inf}{\frac{1}{N} \sum\limits_{i\in\mathfrak{N}}{\Gamma^i_t}}=0$$

- $\zeta,~\gamma,~\beta$: scalar cost parameters which are identical for agents within the same sub-population. 

- $\pi$: the proportion of each sub-population: $\pi^k=\frac{|\mathfrak{N}^k|}{\sum\limits_{j \in \mathcal{K}}{|\mathfrak{N}^j|}}.$

And their values are given in the following table:

|        |$\pi^k$ | $h^k$ | $\sigma^k$ | $\zeta^k$ | $\gamma^k$ | $v^k$ | $\eta^k$ | $\beta^k$ |
| :---:  | :----: | :---: | :--------: | :-------: | :--------: | :---: | :------: | :--------:|
|   k=1  | 0.25   | 0.2   |  0.1       |   1.75    |   1.25     |  0.6  |  0.1     | 1.0       |
|   k=2  | 0.75   | 0.5   |  0.15      |   1.25    |   1.75     |  0.2  |  0.1     | 1.0       |


The framework above can be extended to more realistic models with more than 2 sub-populations and compliance periods, with penalty approximated by multi-knot functions.

## 2. Algorithm And Numeric Tricks

### 2.1. Numeric Tricks
After discretizing the processes in a fine time grid and parameterizing the drifts as well as the initial values, we implement the __*"shooting method"*__ with _**Deep Solvers**_ [(Han, J., Long, J., 2020)](https://doi.org/10.1186/s41546-020-00047-w)[^7] to solve the coupled FBSDEs above. However, the indicator functions in _terminal conditions_ may be tricky to learn directly, so natrually one would use __sigmoid approximation__ to increase continutiy and differentiability. 

$$\mathbf{1}_{0.9>x} \approx \sigma(0.9-x), ~\textit{where the sigmoid function}~\sigma(u)=\frac{1}{1+e^{-u/\delta}}.$$  

In particular, the parameter $\delta$ controls the steepness of $\sigma(\cdot)$ and usually is a small positive number - the smaller $\delta$ is, the more closely it approximates the step of indicator function. On the other hand, the ordinary NN models may learn $V_t^i,U_t^i,Y_t^i \notin [0,1]$ (let's fix $w=1$ for now), which is meaningless as they represent the _probabilities_ of defualting (i.e. missing the quota). And instead of using `tensor.clamp` to forcefully clamp values within $[0,1]$ only, we combine it with the __clamp trick__ to restrict values while maintaining differentiablity. (Same applies to $V_t^i, U_t^i$.)

$$dY_t^i=Y_t^i(1-Y_t^i)Z_tdB_t.$$  

![SigmoidApproximation](Illustration_diagrams/SigmoidApprox.png)
*Smaller $\delta$ leads to closer approximation*

Nonetheless, both the sigmoid approximation and the clamp trick pose huge challenges to the numeric stability. For the sigmoid function, when $\delta$ is too small, there is a great potential for numerical overflow - the exponents could be tremendous especially when $X_t$ is far greater than 0.9, such that `torch.exp(u)==inf` when $u \ge 7.1$. This will raise errors/warnings[^8] in PyTorch. For the clamp trick to work, we must ensure the initial values strictly fall in $(0,1)$. Thus we propose __logit trick__ to map the range $[0,1] \to \mathbb{R}$, which also avoids working with large exponents:

$$
\tilde{Y} := w*\text{logit} (Y/w) = w*\ln\left(\frac{Y/w}{1-Y/w}\right)=f(Y)~.
$$Then apply $\textit{It}\hat{o}  \textit{'s formula}$ (with superscript $[\cdot]^i$ omiited):
$$
\begin{aligned} 
    d \tilde{Y}_t &= (w/2-Y_t)Z_t^2dt + wZ_tdB_t~.\\
\end{aligned}
$$ 

Correspondingly, we use [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) as the loss function, which combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, it takes advantage of the log-sum-exp trick for numerical stability.

Worth mentioning, we experimented with multiple combinations of tricks and loss functions, paired with different optimizers and schedulers. Eventually we chose [Adamax](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html) and [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html) due to their relatively better and more stable performance for all cases in general. Specifically, there are 4 valid combinations of tricks and loss functions[^9]:

```python
{target_type: 'indicator', trick: 'logit', loss_type: 'BCEWithLogitsLoss'}  ## combo 1
{target_type: 'indicator', trick: 'clamp', loss_type: 'BCELoss'}            ## combo 2
{target_type: 'indicator', trick: 'clamp', loss_type: 'MSELoss'}            ## combo 3
{target_type: 'sigmoid'  , trick: 'clamp', loss_type: 'MSELoss'}            ## combo 4
```
More details can be found in the [README](../2Period/Joint_Optim_2Prdx1/README.md) file of 2-agent-2-period scenario. 

### 2.2. Algorithms: Joint-Optimization Vs. Separate-Optimization

As a benchmark to the jointly optimized 2-period model in _1.2._, we also run the 1-period case twice, i.e. minimize the agents' costs in either period separately. Intuitively, the former algorithm can be interpreted as a long-term perspective, considering the future compliance in the current period and thus planning ahead by investing more in increasing their capacities, even when at the first period end. And the latter one can be seen as a short-sighted approach, caring only for the current quota. These 2 distinctive perspectives can make a huge difference in not only the agents' only position, but also the market prices.git  


## 3. Results

To evaluate and visualize the algorithm performances, we define a well-wrapped class `Plot`(:bulb:See more details in [README](../2Period/Joint_Optim_2Prdx1/README.md)), which prodeuces the following plotted results:

- __Agents' behaviours and market impacts__
    - Learnt optimal control processes 
    - Decomposed inventory accumulation pocesses 
    - Inventories in stock during 2 compliance periods
    - Terminal inventories ready-to-submit
    - Market-clearing prices 
- __Algorithm convergency and learning loss__
    - Average forward losses against number of epochs trained
    - Learnt terminal conditions vs. targtes

And here are some example diagramas by 






[^1]: Bellman, R. E.: Dynamic Programming. Princeton University Press, USA (1957).
[^2]: At a finite set of joint points, the posiible lack of differentiability will not have any significant affects.
[^3]: The choice of knot point is associated with $h^{k}$ and total time span $T_1$, $T_2$. A good target (or quota) should be __"attainable"__ - neither too easy nor too hard to achieve. Specifically, even if agents do nothing at all, they will have an initial amount plus a baseline generation of inventories - for instance, $0.2*1 + 0.6=0.8$ for agents in sub-population 1 at the first period end. Similarly, for sub-population 2, all agents will also have a _"garanteed"_ level of 0.8 for delivery. Thus a target reasonably higher than that, i.e. 0.9, would be regard __"attainable"__. 
[^4]: See [_Report-StepwiseDetail_](../FinalReports/Report-StepwiseDetail.md) for more math details.
[^5]: The incremental capacity over baseline can be carried forward to the future periods. 
[^6]: While trading rate may be positive or negative, expansion and overtime-generation rates must be positive.
[^7]: Han, J., Long, J. Convergence of the deep BSDE method for coupled FBSDEs. Probab Uncertain Quant Risk 5, 5 (2020).
[^8]: Examples of [RuntimeError](https://discuss.pytorch.org/t/second-order-derivative-with-nan-value-runtimeerror-function-sigmoidbackwardbackward0-returned-nan-values-in-its-0th-output/173260) and [RuntimeWarning](https://discuss.pytorch.org/t/output-overflow-and-unstablity-when-use-model-eval/3668) on PyTorch Forums. 
[^9]: The indicator target with MSELoss and BCELoss are benchmark models. 