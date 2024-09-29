# Principle-Agent Mean Field Game In Renewable Energy Certificate Market (PA-MFG in REC Markets)
---
Conventional numerical solvers are hard pressed to solve PA-MFG with market-clearing conditions, which may be faced with the "curse of dimentionality". Thus in their study [[1]]("https://doi.org/10.48550/arXiv.2110.01127"), Professor Campbell and his fellows proposed an actor-critic approach to optimization, where the agents form a Nash equilibria according to the principal’s penalty function, and the principal evaluates the resulting equilibria. And they applies this approach to a stylized PA problem arising in Renewable Energy Certificate (REC) markets, where agents may _work_ overtime (or _rent_ capacity), _trade_ RECs, and _expand_ their long-term capacity to navigate the market at maximum profit.

And beyond the origianl study, in the summer of 2024, we further complicated the topic, i.e. extending from 1-period scenario to 2-period scenarios, providing insteresting insights into how "planning ahead" would make a difference in the agents' performance and the markets. 


In this report, we will go throgh the following parts, yet with more weights put on `part 4`, the extended topic to [[1]]("https://doi.org/10.48550/arXiv.2110.01127").

> `Part 1` __FBSDE: Modeling of The PA Problem in REC Markets__
>>    `1.1` REC Market Basics
>>    `1.2` REC Market Modeling: Solving PA Problem Through FBSDE
> `Part 2` __Multi-Step NN Solver: Single-Agent Market in A Single Period__
>>    `2.1` A Simply Example: Smooth Penalty Function
>>    `2.2` Approximation by Linear Combos of Call/Put Functions And Tricks: Improve The Numerical Stability And Convergency  
>>
> `Part 3` __2(multi)-Agent Market in A Single Period__ 
> 
> `Part 4` __2(multi)-Agent Market in 2(multi)-Period__  
>>    `4.1` Joint Optimization: Planning Ahead 
>>    `4.2` Separate Optimization: Considering Present Only 
>>    `4.3` Comparison: Long/Short-Sighted Performances And Market Impacts

---

## `Part 1` FBSDE: Modeling of The PA Problem in REC Markets 

### `1.1` REC Market Basics

Closely related to carbon cap-and-trade (C&T) markets, REC markets are a type of market-based emmissions regulation policies, which are motivating real-world applications of FBSDE in modeling PA-MFG.

In RES markets, a regulator plays the role of principle, setting a floor on the amount of energy generated from renewable resources (aka. green energy) for each firm (based on a percentage of their total production), and providing certificates for each MWh of green energy generated and delivered to the grid. These certificates can be further traded by individual or companies, i.e. agents, to: 1) reduce costs or the greenhouse gas (GHG) emissions impact of their operations; and 2) earn profits from the extra inventories instead of wasting. Since the certificates are traded assets, energy suppliers can trade off between producing clean electricity themselves, and purchasing the certificates on the market. In all, such policies have played an important role in funding clean energy development, particularly in past years when the cost of green power production was not as competitive with the cost of fossil fuel power. 

To ensure compliance, however, each firm must surrender RECs totaling the floor at the end of a compliance period, with a monetary penalty paid for each lacking certificate. And in practice, these systems regulate multiple consecutive and disjoint compliance periods, which are linked together through a mechanism called _banking_, where unused allowances in current period can be carried on to the next period (or multiple future periods). Thus, as an extension to the single-period framework [[1]]("https://doi.org/10.48550/arXiv.2110.01127"), we now consider a 2-period model in this report.[^1]. 

### `1.2` REC Market Modeling: Solving PA Problem Through FBSDE

As is specified above, the 2 compliance periods $[0,T_1]$ and $[T_1,T_2]$ can be denoted as $\mathfrak{T_1}$ and $\mathfrak{T_2}$, respectively. And $T_2$ can be thought of as 'the end of the world', after when there are no costs occurs and all agents forfeit any remaining RECs. Other key notations/parameters used are as follows:

- $i \in \mathfrak{N}$: an individual agent belonging to the whole population $\mathfrak{N}$, annotated by superscript $\cdot^{i}$.

- $k \in \mathcal{K}$: a sub-population of agents, within which all individuals are assumed to have identical preferences and similar initial conditions/capacities, yet across which are distinct. The sub-population is annotated by superscript $\cdot^{k}$.

- $X_t := (X)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the inventories in stock. For some key time points:
    - at $t=0$, there may be some stochastics in the initial inventories, which yet are assumed to be normally distributed. For a specific sub-population $k \in \mathcal{K}$, $X_0^{k} \sim \mathcal{N}(v^k, \eta^k)$.
    - at $t=T_1$, the terminal RECs pre-submission are $X_{T_1}$ carried over from the first period. After forfeiting a minimum amount of $\min\Big(K,X^i_{T_1}\Big)$ inventories, the leftover amounts in stock are: $ReLU\Big(X^i_{T_1}-K\Big)$, which are treated as new initial values for the second period.
    - at $t=T_2$, the terminal RECs pre-submission are $X_{T_2}$.

- $K$: the quota that agents must meet at the end of each period. Any lacking RECs below this floor will be subjected to monetary penalties.

- $P(\cdot)$: a generic penalty function chosen by the regulator, which is assumed continuously differentiable for simplicity. 

- $h$: the baseline generation rate at which agents generate with zero marginal cost. 

- $C_t := (C)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: incremental REC capacity of agents, i.e. the increase of baseline generation rate over time, accumulated by investing in expansion plans - for instance, by installing more solar panels[^2]. 

- $a_t := (a)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the control of expansion rate, representing long-term REC capacity added per unit time. Note that it could be made even more realistic by incorporating a _delay_ between the decision to expand ($a_t$) and the increase to the baseline rate ($h$).

- $g_t := (g)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the control of overtime-generation rate, i.e. the extra capacity achieved by working extra hours and/or renting short-term REC generation capacity at an assumed quadratic cost - specifically, overhour bonus and/or rental fee.

- $\Gamma_t := (\Gamma)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the control of trading rate, with negative[^3] values being the amount sold whereas postive purchased per unit time.

- $S_t := (S)_{t\in\mathfrak{T_1} \cup \mathfrak{T_2}}$: the equilibrium REC price obtained endogenounsly through market-clearing condition: 
$$\lim\limits_{N \to \inf}{\frac{1}{N} \sum\limits_{i\in\mathfrak{N}}{\Gamma^i_t}}=0$$

- $\zeta,~\gamma,~\beta$: scalar cost parameters which are identical for agents within the same sub-population. 

And their values are given in the following table:

|        |$\pi_k$ | $h^k$ | $\sigma^k$ | $\zeta^k$ | $\gamma^k$ | $v^k$ | $\eta^k$ | $\beta^k$ |
| :---:  | :----: | :---: | :--------: | :-------: | :--------: | :---: | :------: | :--------:|
|   k=1  | 0.25   | 0.2   |  0.1       |   1.75    |   1.25     |  0.6  |  0.1     | 1.0       |
|   k=2  | 0.75   | 0.5   |  0.15      |   1.25    |   1.75     |  0.2  |  0.1     | 1.0       |


We consider the agents' problem with $N \to \inf$ agents in total, which implies that all agents are minor entities, having no market impact individually. We work on the filtered probability space $(\Omega,~\mathcal{F},~(\mathcal{F}_{t\in \mathfrak{T}}),~\mathbb{P})$. All processes are assumed to be $\mathcal{F}$-adapted and all controls are associated with quadratic costs. So for agent $i$ in sub-population $k$, the total cost that it seeks to minimize in 2 compliance periods is:

$$
\mathcal{J}^i=\mathbb{E}\Big[
        \Big(
            \int_0^{T_2}{
                \frac{\zeta ^k}{2} (g^i_{\tau})^2 + \frac{\gamma ^k}{2} (\Gamma^i_{\tau})^2 + \frac{\beta ^k}{2} (a^i_{\tau})^2 + S_{\tau}\Gamma_{\tau}
                d\tau} + 
            P\Big(X^i_{T_1}\Big) + 
            P\Big(X^i_{T_2}\Big)
            \Big)
        \Big] ~.
$$

Here, agent $i$ needs to keep track of both its inventories in stock and incremental capacity over time:
$$
\begin{cases}
d X_t^i &= \left(h^k+g_t^i + \Gamma_t^i+ C_t^i \right)dt + \sigma dW_t &&&,~~X_0^i \sim \mathcal{N}\left(v^k, \eta^k\right)\\
dC_t^i &= a_t^i dt &&&,~~C_0^i=0
\end{cases}
$$

Before moving on, we fisrt formulate an assumption: any continuous function $P:\mathbb{R} \to \mathbb{P}$ can be approximated by the linear combination of call/put option payoffs (i.e. shifted and/or scaled ReLU functions):

$$
P(x)=\Phi_0+\sum_{j=1}^{n}{w_j\left(x-K_j\right)_+}~,~~\textit{or}~~P(x)=\Phi_0+\sum_{j=1}^{n}{w_j\left(K_j-x \right)_+}~,
$$

for _weights_ $\Phi_0 \in \mathbb{R},~w_j \in \mathbb{R}_+$ and _knot points_[^5] $K_j \in \mathbb{R}_+ $. Thus any given penalty structure $P$ can be modeled by a multi-knot function. _In future topics/steps_, we will consider a richer class of penalty functions from the principle's perspective, searching for the optimal penalty structure. Yet in this report, we only discuss a simplified case - _**single-knot penalty functions**_ - first fixing the non-compliance function $P$ to a single-knot function with knot $K=0.9$[^6] and intercept $\Phi_0=0$. Then, by tuning the weight $w$, we can see the relation between the penalty level (controled by $w$) and the agents' behaviour, as well as its market impact, i.e.: $P(x)=w(0.9-x)_+ ~, ~~ w=0.25,~0.5,~0.75,~1.0 $ . [^7] Then we proceed into the following intuitive partial proof of the optimization problem above. 

__*Partial Proof*__ Intuitively, we partially differentiate the objective cost function $\mathcal{J}^i( g^i, \Gamma^i, a^i; X^i_{T_1}, X^i_{T_2})$ w.r.t. each control ($g^i$, $\Gamma^i$, $a^i$) in an arbitrary perturbation direction $\eta$, in order to find the optimal controls where the partial derivatives equal to zero. So we first get the partial derivates of $X^i$ w.r.t controls  $g^i,~ \Gamma^i,~ a^i$. For a fixed control $g$, we consider an adapted process $\eta = (\eta_t)_{t≥0}$ and perturb $g$ by $\epsilon> 0$ in the direction of $\eta$: $g+ \epsilon \eta$. Differentiate $X$[^8] in the direction $\eta$:

$$
\begin{align}
    \partial_{g} X_t &= \int_0^t {\eta_s ds}\\
    \partial_{\Gamma} X_t &= \int_0^t {\eta_s ds}\\
    \partial_{a} X_t &= \int_0^t {\partial_{a} C_s ds} = \int_0^t {\int_0^s {\eta_v dv} ds}
\end{align}
$$

Then, differentiate $\mathcal{J}$ w.r.t. $g$ in the direction $\eta$ (same goes for $\Gamma$ and $a$):

$$
\begin{align}
    \partial_{g}\mathcal{J} &= \mathbb{E}\left[\int_0^{T_2}{ {g_s}\eta_s ds+P'(X_T^{0,x})\int_0^T{\eta_s ds}}\right] \nonumber\\ 
    &= \mathbb{E}\left[\int_0^T{\left[a_s+g'(X_T^{0,x})\right]}\eta_s ds\right]  
            &&& \textit{(By Fubini’s Theorem and Iterated Conditioning)} \nonumber\\            
    &= \int_0^T{\mathbb{E}\left[ \left(a_s+P'(X_T^{0,x})\right) \eta_s  \right]ds} \nonumber\\ 
    &= \int_0^T{\mathbb{E}\left[\mathbb{E}\left[ \left(a_s+P'(X_T^{0,x})\right) \eta_s |\mathcal{F}_s\right]\right]ds} \nonumber\\
    &= \mathbb{E}\left[ \int_0^T{\mathbb{E}\left[ \left(a_s+P'(X_T^{0,x})\right) \eta_s |\mathcal{F}_s\right]ds}\right]
            &&& \textit{(Taking out what is known)} \nonumber \\
    &= \mathbb{E}\left[ \int_0^T{\left[ a_s+\mathbb{E}\left[P'(X_T^{0,x})|\mathcal{F}_s\right] \right] \eta_sds}\right]=0 
\end{align}
$$

To minimize $F$ over adapted $a$, we solve for $a$ satisfying the first order condition __(###)__ above for __all__ adapted $\eta$. And it's possible to show that (###) holds __if and only if__ $a_s=-\mathbb{E}\left[P'(X_T^{0,x})|\mathcal{F}_s\right]$ almost surely for almost every $s \in [0,T]$, which implies that $a$ is a martingale. By pplying the Martingale Representation Theorem we get that there exists an $a_0 = −\mathbb{E}\left[P'(X_T^{0,x})\right]$ and adapted $Z = (Z_t)_{t\ge 0}$ such that:

$$a_t=a_0+\int_0^T{Z_s dB_s},~~a_T = −g'(X_T^{0,x}) $$

This conclusion can be extended to the more complicated prob·lem we initially intended to solve. Following the steps of [[1]]("https://doi.org/10.48550/arXiv.2110.01127"), or the probabilistic approach espoused by Professor Carmona and Delarue in [[2]](https://arxiv.org/abs/1210.5780), we can show that the solution for agent $i$ in sub-population $k~(\forall~i \in \mathfrak{N}_k,~k\in\mathcal{K})$, can be found through the solution to the following FBSDE:

$$
\begin{cases}
d X_t^i &= \left( 
                h_t^{k} - 
                \left( 
                    \frac{1}{\zeta^{k}} + \frac{1}{\gamma^{k}}
                    \right) Y_t^{i,X} -
                
                \right)  ,  & X_0 \sim \mathcal{N}(v^k,\eta^k) \\
d a_t = Z_t dB_t, &a_T= −g'(X_T^{0,x})=-2X_T^{0,x}
\end{cases}
$$


<!-- ---  -->
[^1]: Note, same methodology applies to multi-period scenarios.
[^2]: The incremental capacity over baseline can be carried forward to the future periods. 
[^5]: At a finite set of joint points, the posiible lack of differentiability will not have any significant affects. 
[^3]: While trading rate may be positive or negative, expansion and overtime-generation rates must be positive.
[^6]: The choice of knot point is associated with $h^{k}$ and total time span $T_1$, $T_2$. A good target (or quota) should be __"attainable"__ - neither too easy nor too hard to achieve. Specifically, even if agents do nothing at all, they will have an initial amount plus a baseline generation of inventories - for instance, $0.2*1 + 0.6=0.8$ for agents in sub-population 1 at the first period end. Similarly, for sub-population 2, all agents will also have a _"garanteed"_ level of 0.8 for delivery. Thus a target reasonably higher than that, i.e. 0.9, would be regard __"attainable"__. 
[^7]: $w$ could also take any other positive values.
[^8]: The superscript $\cdot^i$ is omitted here for convenience. Same might go for other processes in vicinity. 
---
# Question Log
1. Did we actually only approximated the __*agents' problem*__ through FBSDE, while kept the principle's problem untouched?
