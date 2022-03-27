# Part III Generalized Linear Models
## 1. the exponential family
A class of distributions is in the exponential family if it can be written in the form:
$$
p(y;\eta)=b(y)\exp(\eta^TT(y)-a(\eta))
$$
|||
|-|-|
|$\eta$|natural parameter(canomical parameter)|
|$T(y)$|sufficient statistic(often $T(y)=y$|
|$a(\eta)$|logpartition function|
### 1.1 Bernoulli distribution:
$$
p(y,\phi)=\phi^y(1-\phi)^{1-y}=\exp\left(y\log\left(\frac{\phi}{1-\phi}\right)+\log(1-\phi)\right)
$$
We have:
$$
\begin{aligned}
T(y)&=y\\
\phi&=\frac{1}{1+e^\eta} \\
a(\eta)&=\log(1+e^{\eta})\\
b(y)&=1
\end{aligned}
$$
### 1.2 Gaussian distribution
Let's set $\sigma^2 = 1$
$$
p(y,\mu)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}y^2\right)\exp\left(\mu{y}-\frac{1}{2}\mu^2\right)
$$
We have:
$$
\begin{aligned}
T(y)&=y\\
\eta&=\mu\\
a(\eta)&=\frac{\eta^2}{2}\\
b(y)&=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{y^2}{2}
\right)
\end{aligned}
$$
### 1.3 Other distributions
Real-Gaussian
Binary-Bernoulli
Count-Poisson
$R^+$-Gamma,exponential
distribution over probablilties-beta Dirichlet
## 2 Constructing GLMS
3 Assumptions:
1. $y|x;\theta\sim\text{ExponentialFamily}{(\eta)}$
2. $h(x)=E[T(y)|x]$
3. $\eta=\theta^Tx$ or if $\eta$ is vector-valued, then $\eta_i=\theta_i^Tx$.

Properties:
$$
E[y;\eta]=\frac{\partial}{\partial\eta}a(\eta)\\
E[y;\eta]=\frac{\partial^2}{\partial\eta^2}a(\eta)\\
$$

## 2.1 Ordinary Least Squares
$$ h_\theta(x)=E[y|x;\theta]=\mu=\eta=\theta^Tx$$
## 2.2 Logistic Regression
$$ h_\theta(x)=E[y|x;\theta]=\frac{1}{1+e^{-\eta}}=\frac{1}{1+e^{-\theta^Tx}}$$

**Canonical response function** : $g(\eta)=E[T(y),\eta]$

**Canonical link functoin** : $g^{-1}$
## 2.3 Softmax Regression
Consider a classification problem in which the response variable $y \in \{1,2,\cdots,k \}$. We can use $k$ parameters :
$$
p(y=i,\phi)=\phi_i\quad(\sum_{i=1}^k\phi_i=1)
$$
Then we will instead use $k-1$ parameters $\phi_1,\phi_2,\cdots,\phi_{k-1}$, and $\phi_k=1-\sum_{i=1}^{k-1}\phi_i$, which is not a parameter.

We define $T(y)\in\mathbb{R}^{k-1}$:
$$
T(1)=\left[\begin{matrix}1\\0\\0\\\vdots\\0\end{matrix}\right],
T(2)=\left[\begin{matrix}0\\1\\0\\\vdots\\0\end{matrix}\right],
T(3)=\left[\begin{matrix}0\\0\\3\\\vdots\\0\end{matrix}\right],
T(k-1)=\left[\begin{matrix}0\\0\\0\\\vdots\\1\end{matrix}\right],
T(k)=\left[\begin{matrix}0\\0\\0\\\vdots\\0\end{matrix}\right],
$$
And we define the function $1\{\cdot\}$ that $1\{\text{true}\}=1$ and $1\{\text{false}\}=0$

Now we can show the multinomial is a member of the exponential family
$$
p(y;\phi)=\phi_1^{1\{y=1\}}\phi_2^{1\{y=2\}}\dots\phi_k^{1\{y=k\}}=\cdots=b(y)\exp(\eta T(y)-a(\eta))
$$
where
$$
\begin{aligned}
\eta&=\left[\begin{matrix}\log{\phi_1/\phi_k}\\\log{\phi_2/\phi_k}\\\vdots\\\log{\phi_{k-1}/\phi_k}\\\end{matrix}\right]\\
a(\eta)&=-\log(\phi_k)\\
b(y)&=1
\end{aligned}
$$
the link function:
$$
\eta_i=\log\frac{\phi_i}{\phi_k}
$$
and we can derive that:
$$
\phi_k\sum_{i=1}^ke^{\eta_i}=\sum_{i=1}^k\phi_i=1
$$
We use the assumption 3 that $\eta_i=\theta_i^T x$ $(i=1,2,\dots,k-1)$,and $\theta_k = 0$
$$
p(y=i|x;\theta)=\phi_i=\frac{e^{\eta_i}}{\sum_{j=1}^ke^{\eta_j}}=\frac{e^{\theta_i^Tx}}{\sum_{j=1}^ke^{\theta_j^Tx}}
$$
Our hypothesis will output:
$$
h_\theta(x)=E[T(y)|x;\theta]=\left[\begin{matrix}
    \frac{\exp{\theta_1^Tx}}{\sum_{j=1}^k\exp{\theta_j^Tx}}\\
    \frac{\exp{\theta_2^Tx}}{\sum_{j=1}^k\exp{\theta_j^Tx}}\\
    \vdots\\
    \frac{\exp{\theta_{k-1}^Tx}}{\sum_{j=1}^k\exp{\theta_j^Tx}}
\end{matrix}\right]
$$
and the log-likelihood
$$
l(\theta)=\sum_{i=1}^n\log\prod_{l=1}^k\left(\frac{e^{\theta_l^Tx^{(i)}}}{\sum_{j=1}^ke^{\theta_j^Tx^{(i)}}}\right)^{1 \{y^{(i)}=l\} }
$$