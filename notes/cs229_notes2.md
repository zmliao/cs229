# Part II Classification and logistic regression
We focus on the **binary classification** in which $y\in\{0,1\}$. $1$ is called **positive class** and $0$ the **negative class**
## 1.logistic regression
### 1.1 logistic function
To make $h(x) \in [0,1]$, we choose
$$
h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$
where
$$
g(z)=\frac{1}{1+e^{-z}}
$$
is called the **logistic function** or the **sigmoid function**.
1. $\lim\limits_{z\to\infty}g(z)=1\quad\lim\limits_{z\to-\infty}g(z)=0$
2. $g'(z)=g(z)(1-g(z))$
### 1.2 logistic regression
Let us assume that:
$$
P(y=1|x;\theta)=h_\theta(x) \\
P(y=0|x,\theta)=h_\theta(x) \\
$$
More compactly:
$$
P(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$
Likelihood of the parameters:
$$
L(\theta)=p(\vec{y}|X;\theta)=\prod_{i=1}^n(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
$$
It will be easier to maximize the log likelihood:
$$
l(\theta)=\log{L(\theta)}=\sum_{i=1}^ny^{(i)}\log{h(x^{(i)})}+(1-y^{(i)})\log(1-h(x^{(i)}))
$$
$$
\frac{\partial}{\partial{\theta_j}}l(\theta)=(y-h_\theta(x))x_j.
$$
This give us the stochastic gradient ascent rule:
$$
\theta_j:=\theta_jh+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$
## 2 Perceptron learning algorithm
Change the defination g to be the threshold function:
$$
g(z)=
\begin{cases}
1 &\text{if}&z\geq 0\\
0 &\text{if}&z<0\\
\end{cases}
$$
Let $h_\theta(x)=g(\theta^Tx)$.Use the update rule:
$$
\theta_j:=\theta_jh+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$
then we have the perceptron learning algorithm.
## 3 Newton's method.
Return to logistic regresion.
Suppose we have some function $f:\mathbb{R}\mapsto\mathbb{R}$, and we wish to find a value of $\theta \in \mathbb{R}$ so that $f(\theta) = 0$.Newton's method performs the following update:
$$
\theta:=\theta-\frac{f(\theta)}{f'(\theta)}
$$
If we want to maximize $l(\theta)$, the maxima of l correspondd to points where $l'(\theta)=0$.We can use the update rule:
$$
\theta:=\theta-\frac{l'(\theta)}{l''(\theta)}
$$
In our logistic regression setting ,$\theta$ is vector-valued, we need to generalize Newton's method to this setting:
$$
\theta:=\theta-H^{-1}\nabla_\theta{l(\theta)}
$$
$H$ is called the **Hessian**, whose entries are given by:
$$
H_{ij}=\frac{\partial^2l(\theta)}{\partial{\theta_i}\partial{\theta_j}}
$$
Newton's method enjoys quadratic convergence, faster than gradient descent and requiring fewer iterations to get very close to the minimum.but so long as $d$ is not too large.

When Newton’s method is applied to maximize the logistic regression log likelihood function $l(\theta)$, the resulting method is also called **Fisher scoring**.