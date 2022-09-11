## EM算法以及GMM高斯混合模型

假设有数据集$D=\{ x_i \} _{i=1}^{n}$，数据集中的样本点$x_i,i=1 \cdots n$分别是从K个高斯分布中采样，为了清晰描述这个过程，引入一个随机变量$Z$来描述，它服从概率分布$Q$，其取值为$k\in\{ 1,\cdots,K \}$，令$\alpha(k)=P(Z=k)$，第K个高斯分布可以写为$N(x|Z=K; \overrightarrow{u}_k,\Sigma_k)$

### EM算法对带有隐变量的模型进行参数估计

**EM算法延续了MLE的思路, 通过不断地构造对数似然函数的下界函数, 并对这个较为容易求解的下界函数进行最优化, 以增大对数似然函数取值的下界, 使得在不断的迭代操作后, 对数似然函数的取值能逼近最大值, 从而完成参数的估计**

对数似然函数可以写为：
$$
\begin{aligned}
LL(\theta)&=\log\prod_{i=1}^{n}P(x_i;\theta)\\
&=\sum_{i=1}^{n}\log P(x_i;\theta)\\
&=\sum_{i=1}^{n}\log \sum_{z_i}P(x_i,z_i;\theta)\\
&=\sum_{i=1}^{n}\log \sum_{z_i}[P(z_i|x_i;\theta_{t-1})\frac{P(x_i,z_i;\theta)}{P(z_i|x_i;\theta_{t-1})}]  利用Jensen不等式\\ 
&\ge\sum_{i=1}^{n}\sum_{z_i}[P(z_i|x_i;\theta_{t-1})\log \frac{P(x_i,z_i;\theta)}{P(z_i|x_i;\theta_{t-1})}]
\end{aligned}
$$
不妨将这里最后的下界函数记为，不难看出，最关键的点是需要计算出隐变量的后验概率分布$P(z_i|x_i;\theta_{t-1})$

因此可以将EM算法分为两步:
step1：借助t-1次迭代的参数估计值$\theta_{t-1}$，构造下界函数$B(\theta,\theta_{t-1})$

step2：通过最大化下界函数求得当前迭代的最优$\theta_t$
$$
\theta_t=\text {argmax} \: B(\theta,\theta_{t-1})
$$

### 收敛性

为了保证收敛，需要满足
$$
LL(\theta_t)\ge LL(\theta_{t-1})
$$
对于式(1)，当$\theta=\theta_{t-1}$时，可以取到等号，则
$$
B(\theta_{t-1},\theta_{t-1})=LL(\theta_{t-1})
$$
而当$\theta=\theta_{t}$时，
$$
LL(\theta_t)\ge B(\theta_{t},\theta_{t-1})
$$
而由于$\theta_t=\text {argmax} \: B(\theta,\theta_{t-1})$，则可知
$$
LL(\theta_t)\ge LL(\theta_{t-1})
$$

### EM算法对GMM模型进行求解

首先计算出隐变量的后验概率分布
$$
\begin{aligned}

P(z_i=k|x_i;\theta_{t-1})&=\frac{P(x_i|z_i=k;\theta_{t-1})*P(z_i=k;\theta_{t-1})}{\sum_{z_i}P(x_i|z_i=k;\theta_{t-1})*P(z_i=k;\theta_{t-1})}\\
&=\frac{\alpha(k)_{t-1}N(x_i;\overrightarrow{u}_k,\Sigma_k)}{\sum_{k}\alpha(k)_{t-1}N(x_i|\overrightarrow{u}_k,\Sigma_k)}\\
&=q_{ik}

\end{aligned}
$$
则相应的下界函数可以写为
$$
\begin{aligned}

B(\theta,\theta_{t-1})&=\sum_{i=1}^{n}\sum_{z_i}[P(z_i|x_i;\theta_{t-1})\log \frac{P(x_i,z_i;\theta)}{P(z_i|x_i;\theta_{t-1})}]\\
&=\sum_{i=1}^{n}\sum_{z_i}[q_{ik}\log \frac{P(x_i|z_i;\theta)*P(z_i;\theta)}{q_{ik}}]\\
&=\sum_{i=1}^{n}\sum_{k}[q_{ik}\log \frac{\alpha(k)_{t}N(x_i;\overrightarrow{u}_k,\Sigma_k)}{q_{ik}}]\\
&=\sum_{i=1}^{n}\sum_{k}q_{ik}[\log \alpha(k)_{t} +\log N(x_i;\overrightarrow{u}_k,\Sigma_k) -\log{q_{ik}}]\\

\end{aligned}
$$
其中，
$$
\begin{aligned}
\log N(x_i;\overrightarrow{u}_k,\Sigma_k)&=\log \frac{1}{\sqrt {\det(\mathbf \Sigma_k)}(2\pi)^{\frac n2}}exp(-\frac12(\mathbf x-\mathbf \mu_k)^T\mathbf \Sigma_k^{-1}(\mathbf x-\mathbf \mu_k))\\
&=(-\frac{n}{2}\log 2\pi-\frac{1}{2}\log|\mathbf \Sigma_k|-\frac12(\mathbf x-\mathbf \mu_k)^T\mathbf \Sigma_k^{-1}(\mathbf x-\mathbf \mu_k))
\end{aligned}
$$
那么带入式(8)有
$$
\begin{aligned}

B(\theta,\theta_{t-1})&=\sum_{i=1}^{n}\sum_{k}q_{ik}[\log \alpha(k)_{t} +\log N(x_i;\overrightarrow{u}_k,\Sigma_k) -\log{q_{ik}}]\\
&=\sum_{i=1}^{n}\sum_{k}q_{ik}[\log \alpha(k)_{t} -\frac{n}{2}\log 2\pi-\frac{1}{2}\log|\mathbf \Sigma_k|-\frac12(\mathbf x-\mathbf \mu_k)^T\mathbf \Sigma_k^{-1}(\mathbf x-\mathbf \mu_k) -\log{q_{ik}}]\\


\end{aligned}
$$
去掉其中的常数项，则需要优化的函数可以简化为
$$
\begin{aligned}

B(\theta,\theta_{t-1})&=\sum_{i=1}^{n}\sum_{k}q_{ik}[\log \alpha(k)_{t} -\frac{1}{2}\log|\mathbf \Sigma_k|-\frac12(\mathbf x-\mathbf \mu_k)^T\mathbf \Sigma_k^{-1}(\mathbf x-\mathbf \mu_k) ]\\


\end{aligned}
$$
因此，对式(11)进行求导，
$$
\begin{aligned}

\frac{\partial B(\theta,\theta_{t-1})}{\partial \mu_k}
B(\theta,\theta_{t-1})&=\sum_{i=1}^{n}q_{ik}[\Sigma_k^{-1}(\mathbf x_i-\mathbf \mu_k)]\\
\Sigma^{-1}(\sum_{i=1}q_{ik}x_i-\sum_{i=1}\mu_kq_{ik})=0\\
\mu_k=\frac{\sum_{i=1}q_{ik}x_i}{\sum_{i=1}q_{ik}}



\end{aligned}
$$

$$
\frac{\partial B(\theta,\theta_{t-1})}{\partial \Sigma_k}=\\
\Sigma_k=\frac{\sum_{i=1}^{n} q_{ik}(\mathbf x-\mathbf \mu_k)(\mathbf x-\mathbf \mu_k)^T}{\sum_{i=1}^{n} q_{ik}}
$$

对于$\alpha_k$，由于它是概率分布，和为1，因此是带约束的优化问题，需要采用拉格朗日乘子法，仅考虑与该项相关的部分，
$$
L(\alpha,\lambda)=\sum_{k}(\sum_{i=1}^{n}q_{ik}[\log \alpha(k)])+\lambda(1-\sum_{k}\alpha(k))\\
\frac{\partial L}{\partial \alpha_k}=\sum_{i=1}^{n}q_{ik}\frac{1}{\alpha_k}-\lambda=0\\
得\alpha_k=\frac{1}{\lambda}\sum_{i=1}^{n}q_{ik} \;两边积分\\
\lambda=\sum_{k}(\sum_{i=1}^{n}q_{ik})=n
$$
因此最后带入可得
$$
\alpha_k=\frac{1}{n}\sum_{i=1}^{n}q_{ik}
$$


## 传统变分法

### 变分法的引入-最速降线问题

最早提出最速降线问题的是伽利略，他探讨在只考虑重力的情况下，一个质点从点A运动到比A低的点B，以怎样的路径行进才能使所需的时间最短，这个看似简单的问题，结果并不显然。

要找到一条最短路径，使得1到2的运动时间最短

<img src="EM算法以及GMM、变分法.assets/v2-e209f29bd4e713c0da721a04d7cf1d35_b.jpg" alt="img" style="zoom:50%;" />

要得到总时间$S$，需要对每一段位移进行微分，$S=\int_{1}^{2}(1/v)ds$，根据能量守恒定律，速度可以写为$v=\sqrt{2gy}$，以$x$为自变量描述路径，则距离微元可以表述为
$$
ds=\sqrt{(dx)^2+(dy)^2}=\sqrt{1+y'^2}dx
$$
所以可以得到方程
$$
S=\int_{1}^{2}(1/v)ds=\int_{x_1}^{x_2}\frac{1}{\sqrt{2gy}}\sqrt{1+y'^2}dx=\int_{x_1}^{x_2}f(y,y')dx
$$

### 变分法

对于类似的速降线问题，称之为变分问题，自变量为函数$y(t)$，其对应的函数为$S(y(t))$，将这种映射关系成为泛函关系。

<img src="EM算法以及GMM、变分法.assets/v2-fb52ffbe6c465079cd7ee13937d6169a_b.jpg" alt="img" style="zoom: 50%;" />

将$y(x)$看成一条由$(x_1,y_1)$到$(x_2,y_2)$的曲线，那么$y(x)$对应最优解，虚线代表偏差为$\alpha \eta(x)$的错误解，任意一条曲线满足$Y(x)=y(x)+\alpha\eta(x)$，其中$\alpha$为任意实数，$\eta(x)$为任意函数，并且它需要满足边界条件，也就是
$$
\eta(x_1)=\eta(x_2)=0
$$
此时，引入参数$\alpha$后，泛函可以写为
$$
\begin{aligned}
S(\alpha)&=\int_{x_1}^{x_2}f(Y,Y',x)dx\\
&=\int_{x_1}^{x_2}f(y+\alpha \eta,y'+\alpha \eta',x)dx
\end{aligned}
$$
无论最优解对应的是极大值还是极小值，都有$dS/d\alpha|_{\alpha=0}=0$，因此这里先计算$f$的偏导数
$$
\frac{\partial f}{\partial \alpha}=\eta \frac{\partial f}{\partial Y}+\eta'\frac{\partial f}{\partial Y'}+0\frac{\partial f}{\partial x}
$$
则
$$
\frac{d S(\alpha)}{d \alpha}=\int_{x_1}^{x_2}(\eta \frac{\partial f}{\partial Y}+\eta'\frac{\partial f}{\partial Y'})dx
$$
利用分布积分计算后半部分的积分有
$$
\begin{aligned}
\int_{x_1}^{x_2}\eta'\frac{\partial f}{\partial Y'}dx&=\int_{x_1}^{x_2}\frac{\partial f}{\partial Y'}d\eta\\
&=\eta\frac{\partial f}{\partial Y'}|_{x_1}^{x_2}-\int_{x_1}^{x_2}\eta\frac{d}{dx}\frac{\partial f}{\partial Y'}dx\\
&=-\int_{x_1}^{x_2}\eta\frac{d}{dx}\frac{\partial f}{\partial Y'}dx
\end{aligned}
$$
将式(22)带入式(21)，并根据极值点满足$dS/d\alpha|_{\alpha=0}=0$，有
$$
\begin{aligned}
\frac{d S(\alpha)}{d \alpha}&=\int_{x_1}^{x_2}(\eta \frac{\partial f}{\partial Y}-\eta\frac{d}{dx}\frac{\partial f}{\partial Y'})dx\\带入\alpha=0有
&=\int_{x_1}^{x_2}\eta( \frac{\partial f}{\partial y}-\frac{d}{dx}\frac{\partial f}{\partial y'})dx=0
\end{aligned}
$$
由$\eta$的任意性，最终得到**Euler-Lagrange 方程**
$$
\frac{\partial f}{\partial y}-\frac{d}{dx}\frac{\partial f}{\partial y'}=0
$$

## 平均场变分推断

在平均场理论中，将隐变量$z$做独立划分$z_1、z_2、...、z_M$，假设分布为$q(Z)=\prod_{i=1}^{M}q_i(z_i)$，在EM算法中，最重要的就是用它来拟合隐变量的分布

对数似然函数可以写为
$$
\begin{aligned}
L(q)=\text {log}P(X)&=\text {log}q(Z)\frac{P(X)}{q(Z)}\\
&=\int q(Z)\text{log}q(Z)\frac{P(X)}{q(Z)}dZ\\
&=\int q(Z)\text{log}q(Z)\frac{P(X,Z)}{q(Z)P(Z|X)}dZ\\
&=\int q(Z)\text{log}\frac{q(Z)}{P(Z|X)}dZ+\int q(Z)\text{log}\frac{P(X,Z)}{q(Z)}dZ\\
&=KL(q(Z)||(P(Z|X))+ELBO

\end{aligned}
$$
带入平均场的隐变量分布形式，当模型参数既分布$P$中的参数$\theta$固定时，希望找到隐变量分布的最优估计，及希望KL散度最小，而此时似然值$L(q)$是$\theta$的函数，为值，因此可以等价于找到最大的ELBO，既
$$
\begin{aligned}
\mathop{max}\limits_{q(Z)}\ \int q(Z)\text{log}P(X,Z)dZ-\int q(Z)\text{log}q(Z)dZ&=\mathop{max}\limits_{q(Z)}\ \int \prod_{i=1}^{M}q_i(z_i)\text{log}P(X,Z)dZ-\int \prod_{i=1}^{M}q_i(z_i)\text{log}\prod_{i=1}^{M}q_i(z_i)dZ\\

\end{aligned}
$$
先考虑前半部分，此时为了优化，对某一个固定的$q_j(z_j)$进行考虑，与它不相关的部分可以看做常数：
$$
\begin{aligned}
\int \prod_{i=1}^{M}q_i(z_i)\text{log} P(X,Z)dZ&=\int q_1(z_1)...q_M(z_M)\text{log}(X,z_1...z_M)dz_1...dz_M\\
&=\int q_j(z_j)\int \prod_{i=1,i\ne j}^{M}q_i(z_i) \text{log}(X,z_1...z_M))dz_1...dz_M\\
&=\int q_j(z_j)\mathbb (E_{\prod_{i=1,i\ne j}^{M}q_i(z_i)} \text{log}(X,z_1...z_M))dz_j\\
&=\int q_j(z_j)\mathbb (E_{-j} \text{log}(X,z_1...z_M))dz_j\\

\end{aligned}
$$
为了与式(28)对应，这里将分布的期望表示为
$$
\mathbb (E_{-j} \text{log}(X,z_1...z_M))dz_j=\text{log} \hat{p}(X,z_j)
$$
再考虑后半部分，注意将里面与$q_j(z_j)$无关的项当做常数处理
$$
\begin{aligned}
\int \prod_{i=1}^{M}q_i(z_i)\text{log}\prod_{i=1}^{M}q_i(z_i)dZ&=\int \prod_{i=1}^{M}q_i(z_i)(\sum_{i=1}^{M}\text{log}q_i(z_i))dZ\\
&=\int q_1(z_1)...q_M(z_M)\text{log}q_j(z_j)dz_1...dz_M+constant\\
&=\int q_j(z_j)\text{log}q_j(z_j)dz_j+constant\\

\end{aligned}
$$
因此，(26)的最大化问题等价为
$$
\begin{aligned}
\mathop{max}\limits_{q(z_j)}\ \int q(Z)\text{log}P(X,Z)dZ-\int q(Z)\text{log}q(Z)dZ&=\mathop{max}\limits_{q(z_j)}\ \int \prod_{i=1}^{M}q_i(z_i)\text{log}P(X,Z)dZ-\int \prod_{i=1}^{M}q_i(z_i)\text{log}\prod_{i=1}^{M}q_i(z_i)dZ\\
&=\mathop{max}\limits_{q(z_j)}\int q_j(z_j)\text{log} \hat{p}(X,z_j)dz_j-\int q_j(z_j)\text{log}q_j(z_j)dz_j\\
&=\mathop{max}\limits_{q(z_j)}\int q_j(z_j)\text{log} \frac{\hat{p}(X,z_j)}{q_j(z_j)}dz_j\\
&=\mathop{max}\limits_{q(z_j)}\: -KL(q_j(z_j)||\hat{p}(X,z_j))

\end{aligned}
$$
上述优化问题等价于最小化KL散度，则隐变量的最优分布为$q_j(z_j)=\hat{p}(X,z_j)$，不难看出，这是一种迭代优化的方法，每一次在求$q_j(z_j)$时，都需要使用前面更新过的$q_{j-1}(z_{j-1})...$，这一迭代过程也是**坐标上升法（Coordinate Ascend）**的思想。

但这种经典变分推断，也存在一些问题：

- **假设太强，对复杂模型也许假设不好甚至不成立**
- **即使假设是OK的，但是因为其递推式包含很多积分，可能是无法计算的(Intractable)**

实际上在上面的推导过程中，如果对观测变量$X$施加独立性假设，即满足$P(X)=p(x_1)...p(x_N)$，则$\text{log}P(X,Z)=\Sigma_{i=1}^{N}\text{log}P(x_i,z)$

### 随机梯度变分推断(SGVI)

上一节对隐变量的分布做出了独立性的假设，本节中假定隐变量整体服从一个参数为$\phi$的分布，记为$q_{\phi}(Z)$，同时这里利用观测独立性的假设，只考虑某一个样本的梯度。
$$
\begin{aligned}
\nabla_{\phi} L'(q)&=\nabla_{\phi}ELBO\\
&=\nabla_{\phi}(\int q_{\phi}(Z)\text{log}P(x_i,Z)dZ-\int q_{\phi}(Z)\text{log}q_{\phi}(Z)dZ)\\
&=\int q_{\phi}(Z)\nabla_{\phi}\text{log}q_{\phi}(Z)\text{log}P(x_i,Z)dZ-\int \nabla_{\phi}(q_{\phi}(Z))\text{log}q_{\phi}(Z)dZ-\int q_{\phi}(Z)(\nabla_{\phi}(\text{log}q_{\phi}(Z))dZ\\
&=\int q_{\phi}(Z)\nabla_{\phi}\text{log}q_{\phi}(Z)\text{log}P(x_i,Z)dZ-\int q_{\phi}(Z)\nabla_{\phi}\text{log}q_{\phi}(Z)\text{log}q_{\phi}(Z)dZ-\int \nabla_{\phi}q_{\phi}(Z)dZ\\
&=\int q_{\phi}(Z)\nabla_{\phi}\text{log}q_{\phi}(Z)(P(x_i,Z)-\text{log}q_{\phi}(Z))dZ\\
&=\mathbb E_{q_{\phi}(Z)}[\nabla_{\phi}\text{log}q_{\phi}(Z)(P(x_i,Z)-\text{log}q_{\phi}(Z))]
\end{aligned}
$$
这里在化简时用到了两个关系：
$$
\nabla_{\phi} q_{\phi}(Z)=q_{\phi}(Z)\nabla_{\phi}\text{log} q_{\phi}(Z)\\
\int \nabla_{\phi}q_{\phi}(Z)dZ=\nabla_{\phi}\int q_{\phi}(Z)dZ=0
$$
式(31)的最终形式标明梯度的求解可以转化为采用蒙特卡罗法进行采样，然后取期望即计算平均值，最后用梯度上升法进行参数的迭代更新。

注意式(31)中存在$\nabla_{\phi}\text{log}q_{\phi}(Z)$，而通过蒙特卡罗法进行采样时，有可能采样到使$q_{\phi}(Z)$非常小的点，就可能导致这部分的变动非常不稳定，导致方差很大，即high variance问题

因此改进的主要目标就是high variance，这里采用重参数化技巧(Reparameterization)。如果式(31)第二行求期望的时候，这个分布与$\phi$无关最好，这里假定
$$
z=g_{\phi}(\epsilon,x^{i})\\
\epsilon\sim p(\epsilon)
$$
那么有
$$
q_{\phi}(z|x^{i})dz=p(\epsilon)d\epsilon
$$

$$
\begin{aligned}
\nabla_{\phi} L'(q)&=\nabla_{\phi}(\int q_{\phi}(Z)\text{log}P(x_i,Z)dZ-\int q_{\phi}(Z)\text{log}q_{\phi}(Z)dZ)\\
&=\nabla_{\phi}\int (\text{log}P(x_i,Z)-\text{log}q_{\phi}(Z))q_{\phi}(Z)dZ)\\
&=\nabla_{\phi}\int (\text{log}P(x_i,Z)-\text{log}q_{\phi}(Z))p(\epsilon)d\epsilon\\
&=\mathbb E_{p(\epsilon)}[\nabla_{\phi} (\text{log}P(x_i,Z)-\text{log}q_{\phi}(Z))]\\
&=\mathbb E_{p(\epsilon)}[\nabla_{z} (\text{log}P(x_i,Z)-\text{log}q_{\phi}(Z))\nabla_{\phi}	z]\\
\end{aligned}
$$

因此最后计算完成后，可以转换为对$\epsilon$进行采样，即将随机性都转移到了$\epsilon$上
