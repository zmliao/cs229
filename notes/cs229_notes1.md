# Part I Linear Regression
## 1.Supervised learning
Notation:
|||
|-|-|
|$x^{(i)}$|input variable/ input features|
|$y^{(i)}$|target variable|
|$(x^{(i)},y^{(i)})$|training example|
|$\{(x^{(i)},y^{(i)});i=1,\cdots,n\}$|training set|
|$(i)$|an index into the training set|
|$\mathcal{X}$|the space of input values|
|$\mathcal{Y}$|the space of output values|
|$h:\mathcal{X}\mapsto\mathcal{Y}$|hypothesis|
The procsss of supervised learning:
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20.14.58.png?expiry=1647907200000&hmac=fed8FMr-EiEj2rvoAkecadM16jTShXr3olDQqT8H3Ok)
Regression: Target variable is continuous.
Classification: Target variable is discrete.
## 2.LMS algorithm
### 2.1 hypothesis of linear regression
If $x$ are two-dimensional vectors in $\mathbb{R}^2$, we can decide to approximtate $y$ as a linear function of $x$:
$$
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2
$$
Here the $\theta_i$'s are the **parameters**.

To simplify our notatoin, we also introduce the **intercept term** $x_0=1$,so that
$$
h(x)=\sum_{i=0}^d\theta_ix_i=\theta^Tx
$$
where $\theta$ and $x$ are vectors,and $d$ is the number of input variables.

### 2.2 cost function
To make $h(x)$ close to y, we define a function that measures, for each values of $\theta$'s, how close the  $h(x^{(i)})$'s are to the corrsponding $y(i)$'s:
$$
J(\theta)=\frac{1}{2}\sum_{i=1}^n(h_\theta(x^{(i)})-y^{(i)})^2
$$
That give rise to the **ordinary least squares** regression model.

### 2.3 gradient descent
We want to choose $\theta$ as to minimize$J(\theta)$.

Start with initial guess for $\theta$, and repeatedly performs the update:
$$
\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)
$$
until convergence.Here $\alpha$ is the **learning rate**.

We have:
$$
\frac{\partial}{\partial\theta_j}J(\theta)=(h_\theta(x)-y) x_j
$$
For a single training example,this give the **LMS (least mean squares)** rule or **Widrow-Hoff** learning rule:
$$
\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$
Because $J$ is a convex quadratic function, the optimization problem has only one global, and no other local optima.

There are two ways to modify this method for a training set of more than one expample.
### 2.4 batch gradient descent
Repeat until convergence:
$$
\theta := \theta + \alpha\sum_{i=1}^{n}(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}
$$

This method looks at every example in the **entire** training set on every step. 

### 2.5 stochastic gradient descent

for $i=1$ to $m$, $\theta_j := \theta_j-\alpha(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}$, (for every $j$).

We update the parameters according to the gradient of the error with respect to that **single training example** only.

Schocahstic gradient descent gets $\theta$ "close" to the minimum much faster than batch gradient descent, particularly when the training set is large. Schocahstic gradient descent do not go in the most direct direction downhill,and may never converge the minimum,but $\theta$ will keep oscillating around the minimum of $J(\theta)$.

Mini-batch gradient descent use a small part of the training set on every step.

## 3.The normal equations
### 3.1 Matrix derivatives
For a function $f:\mathbb{R}^{n\times d}\mapsto\mathbb{R}$:
$$
\nabla_Af(A)=\left[
    \begin{matrix}
    &\frac{\partial{f}}{\partial{A_{11}}}&\cdots&\frac{\partial{f}}{\partial{A_{1d}}}\\
    &\vdots&\ddots&\vdots& \\
    &\frac{\partial{f}}{\partial{A_{n1}}} &\cdots&\frac{\partial{f}}{\partial{A_{nd}}}&\\
    \end{matrix}
\right ]
$$
### 3.2 Least squares revisited
**design matrix** $X$:
$$
X=\left[ 
    \begin{matrix}
    & — & {{({{x}^{(1)}})}^{T}} & — & \\
    & — & {{({{x}^{(2)}})}^{T}} & — & \\
    &&\vdots&\\
    & — & {{({{x}^{(n)}})}^{T}} & — & \\
    \end{matrix} 
    \right]
$$
All the target value from the training set:
$$
\vec{y}=\left[
    \begin{matrix}
    y^{(1)}\\
    y^{(2)}\\
    \vdots\\
    y^{(n)}\\
    \end{matrix}
\right]
$$
Since $h_\theta(x^{(i)})=(x^{(i)})^T\theta$
$$
J(\theta)=\frac{1}{2}\sum_{i=1}^n(h_\theta(x^{(i)})-y^{(i)})^2=\frac{1}{2}(X\theta-\vec{y})^T(X\theta-\vec{y})
$$
To minimize J,let's find its derivatives with respect to $\theta$:
$$
\nabla_Af(A)=X^TX\theta-X^T\vec{y}
$$
We set its derivatives to zero, and we obtain the normal equations:
$$
X^TX\theta = X^T\vec{y}
$$
The value of $\theta$ that minimizes $J(\theta)$.
$$
\theta=(X^TX)^{-1}X^T\vec{y}
$$
## 4.Probabilistic interpretation
Assume that:
$$y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}$$
Where $\epsilon^{(i)}$ is an error term distributed IID (independently and identically distributed). We can write $\epsilon^{(i)}\sim\mathcal{N}(0,\sigma^2)$:
$$
p(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)
$$
$$
p(y^{(i)}|x^{(i)},\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y{(i)}-\theta^{T}x^{(i)})^2}{2\sigma^2}\right)
$$
Likelihood function:
$$
L(\theta)=L(\theta;X,\vec{y})=p(\vec{y}|X;\theta)=\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y{(i)}-\theta^{T}x^{(i)})^2}{2\sigma^2}\right)
$$
The pricipal of maximum likelihood says that we should choose $\theta$ to make the data as high probability as possible. To maximize $L(\theta)$,we instead maximize the log likelihood:
$$
l(\theta)=\log{L(\theta)}=n\log{\frac{1}{\sqrt{2\pi}\sigma}}-\frac{1}{2\sigma^2}\sum_{i=1}^n(y^{(i)}-\theta^Tx^{(i)})^2
$$
Hence, maximizing $l(\theta)$ is minimizing:
$$
J(\theta)=\frac{1}{2}\sum_{i=1}^n(y^{(i)}-\theta^Tx^{(i)})^2
$$
That's why we use least-squares cost function.
## 5 Locally weighted linear regression(LWR)
### 5.1 non-parametric algorithm
**Parametric** learning algorithm has fixed, finite number of parameters, which are fit to the data. We no longer need to keep the trainin data to make predictions. The unweighted linear regression algorithm is parametric.

**Non-parametric**:The amount of stuff we need to keep in order to represent the hypothesis grows with the size of the training set. Locally weighted linear regression is non-parametric algorithm.

### 5.2 locally weighted linear regression
To make a prediction at a query point x, the locally weighted linear regression algorithm does the following:
1. Fit $\theta$ to minimize $\sum_i w^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2$
2. output $\theta^Tx$.

$w(i)$'s are non-negative valued weights, a standard choice is:
$$
w^{(i)}=\exp\left(-\frac{(x^{(i)}-x)^2}{2\tau^2}\right)
$$
where $\tau$ is called the **bandwidth**, controls how quickly the weight of a training example falls off with distance of its $x^{(i)}$