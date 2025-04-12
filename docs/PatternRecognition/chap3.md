# 概率密度函数的估计

## 最大似然估计
### 最大似然估计的基本原理
假设样本集包含$N$个样本，即$$\mathscr{X} = \{\bm{x}_1, \bm{x}_2, ..., \bm{x}_N\}$$

由于样本是独立地从$p(\bm{x} | \theta)$中抽取的，因此在概率密度为$p(\bm{x} | \theta)$时，获得样本集$\mathscr{X}$的概率为$$l(\theta) = p(\mathscr{X} | \theta) = p(\bm{x}_1, \bm{x}_2, ..., \bm{x}_N | \theta) = \prod_{i = 1}^N p(\bm{x}_i | \theta)$$

这一函数也被称作**参数$\theta$相对于样本集$\mathscr{X}$的似然函数**。

**最大似然估计量：** 令$l(\theta)$为样本集$\mathscr{X}$的似然函数，$\mathscr{X} = \{\bm{x}_1, \bm{x}_2, ..., \bm{x}_N\}$，如果$\hat{\theta} = d(\mathscr{X}) = d(\bm{x}_1, \bm{x}_2, ..., \bm{x}_N)$是参数空间$\varTheta$中能够使似然函数$l(\theta)$最大化的$\theta$值，则$\hat{\theta}$为$\theta$的最大似然估计量。也记作$$\hat{\theta} = \argmax l(\theta)$$

类似地，可定义**对数似然函数**：$$H(\theta) = \ln l(\theta) = \sum_{i = 1}^N \ln p(l | \theta)$$

### 最大似然估计的求解
+ + 一个待估计参数：$$\dfrac{d l(\theta)}{d \theta} = 0 \quad\text{or}\quad \dfrac{d H(\theta)}{d\theta} = 0$$
+ 多个待估计参数（如$\bm{\theta} = (\theta_1, \theta_2, ..., \theta_s)^T$）：引入梯度算子$$\nabla_{\bm{\theta}} = (\dfrac{\partial}{\partial \theta_1}, \dfrac{\partial}{\partial \theta_2}, ..., \dfrac{\partial}{\partial \theta_s})^T$$使得$$\nabla_{\bm{\theta}} l(\bm{\theta}) = 0 \quad\text{or}\quad \nabla_{\bm{\theta}} H(\bm{\theta}) = \sum_{i = 1}^N \nabla_{\bm{\theta}} \ln p(\bm{x}_i | \bm{\theta}) = 0$$

### 正态分布的最大似然估计
+ 一元正态分布：$$\begin{aligned}
    \hat{\mu} =& \dfrac{1}{N} \sum_{k = 1}^N x_k \\
    \hat{\sigma}^2 =& \dfrac{1}{N} \sum_{k = 1}^N (x_k - \hat{\mu})^2 \\
\end{aligned}$$
    + 注意：$\hat{\sigma}^2$的**无偏估计**实际为$$\hat{\sigma}^2 = \dfrac{1}{N - 1} \sum_{k = 1}^N (x_k - \hat{\mu})^2$$
+ 多元正态分布：$$\begin{aligned}
    \hat{\bm{\mu}} =& \dfrac{1}{N} \sum_{k = 1}^N \bm{x}_k \\
    \hat{\bm{\Sigma}} =& \dfrac{1}{N} \sum_{k = 1}^N (\bm{x}_k - \hat{\bm{\mu}})(\bm{x}_k - \hat{\bm{\mu}})^T \\
\end{aligned}$$
    + 注意：$\hat{\bm{\Sigma}}$的**无偏估计**实际为$$\hat{\bm{\Sigma}} = \dfrac{1}{N - 1} \sum_{k = 1}^N (\bm{x}_k - \hat{\bm{\mu}})(\bm{x}_k - \hat{\bm{\mu}})^T$$

## 贝叶斯估计与贝叶斯学习
与最大似然估计不同：**将待估计参数视为随机变量，而非未知的固定量。**

### 贝叶斯估计
将待估计参数$\bm{\theta}$看作具有先验分布密度$p(\theta)$的随机变量，其取值与样本集$\mathscr{X}$相关。对连续变量$\theta$，将其估计为$\hat{\bm{\theta}}$所带来的损失为$\lambda(\hat{\bm{\theta}}, \theta)$，称为**损失函数**。

设样本的取值空间为$E^d$，参数的取值空间为$\varTheta$，则使用$\hat{\bm{\theta}}$来作为估计时的总**期望风险**就是$$\begin{aligned}
    R =& \int_{E^d}\int_{\varTheta} \lambda(\hat{\bm{\theta}}, \theta) p(\bm{x}, \theta)\, d\theta \, d\bm{x} \\
    =& \int_{E^d}\int_{\varTheta} \lambda(\hat{\bm{\theta}}, \theta) p(\theta | \bm{x}) p(\bm{x})\, d\theta \, d\bm{x} \\
\end{aligned}$$

定义在样本$\bm{x}$下的条件风险为$$R(\hat{\bm{\theta}} | \bm{x}) = \int_{\varTheta} \lambda(\hat{\bm{\theta}}, \theta) p(\theta | \bm{x})\, d\theta$$则期望风险可写成$$R = \int_{E^d}R(\hat{\bm{\theta}} | \bm{x}) p(\bm{x}) \, d\bm{x}$$

注意到条件风险的非负性，求期望风险的最小值就是对所有样本求最小的条件风险。即：
$$\theta^* = \argmin_{\hat{\bm{\theta}}} R(\hat{\bm{\theta}} | \mathscr{X}) = \argmin_{\hat{\bm{\theta}}} \int_{\varTheta} \lambda(\hat{\bm{\theta}}, \theta) p(\theta | \mathscr{X})\, d\theta$$

对于离散的决策分类，我们需要事先定义**决策表（损失表）**；对于连续情况，我们需要定义**损失函数**。一个常用的损失函数为**平方误差损失函数**（Mean Square Error，MSE）：$$\lambda(\hat{\bm{\theta}}, \theta) = (\theta - \hat{\bm{\theta}})^2$$

采用平方误差损失函数，则在样本$\bm{x}$条件下$\theta$的贝叶斯估计量$\theta^*$为$$\theta^* = E[\theta | \bm{x}] = \int_{\varTheta} \theta p (\theta | \bm{x}) \,d\theta$$

给定样本集$\mathscr{X}$，$\theta$的贝叶斯估计量$\theta^*$可类似求得$$\theta^* = E[\theta | \mathscr{X}] = \int_{\varTheta} \theta p (\theta | \mathscr{X}) \,d\theta$$

从而，在采用平方误差损失函数的情况下，贝叶斯估计的步骤为：
1. 根据对问题的认识或者猜测，确定$\theta$的先验分布密度$p(\theta)$。
2. 由于样本是独立同分布的，而且已知样本密度函数的形式$p(\bm{x} | \theta)$，可以在形式上求出样本集的联合分布为$$p(\mathscr{X} | \theta) = \prod_{i = 1}^N p(\bm{x}_i | \theta)$$
3. 利用贝叶斯公式求出$\theta$的后验分布$$p(\theta | \mathscr{X}) = \dfrac{p(\mathscr{X} | \theta) p(\theta)}{\int_{\varTheta}p(\mathscr{X} | \theta) p(\theta)\, d\theta}$$
4. 可以推出$\theta$的贝叶斯估计量$\theta^*$为$$\theta^* = \int_{\varTheta} \theta p(\theta | \mathscr{X}) \, d\theta$$

**注意：** 在求出$p(\theta | \mathscr{X})$之后，可以直接给出样本的概率密度函数：$$p(\bm{x} | \mathscr{X}) = \int_{\varTheta} p(\bm{x} | \theta) p(\theta | \mathscr{X}) \, d\theta$$

理解：$\theta$为随机变量，其拥有一定的分布；而要估计的概率密度$p(\bm
{x} | \mathscr{X})$就是**所有可能的参数取值下的样本概率密度的加权平均**，而**这个加权**就是**在观测样本下估计出的参数$\theta$的后验概率**。

### 贝叶斯学习
将样本集重新记作$\mathscr{X}^N = \{\bm{x}_1, \bm{x}_2, ..., \bm{x}_N\}$，则贝叶斯估计量的式子可重写如下：$$\theta^* = \int_{\varTheta} \theta p(\theta | \mathscr{X}^N) \, d\theta$$其中$$p(\theta | \mathscr{X}^N) = \dfrac{p(\mathscr{X}^N | \theta) p(\theta)}{\int_{\varTheta}p(\mathscr{X}^N | \theta) p(\theta)\, d\theta}$$

又当$N > 1$时，有$$p(\mathscr{X}^N | \theta) = p(\bm{x}_N | \theta)p(\mathscr{X}^{N - 1} | \theta)$$

从而能够得到递推式$$p(\theta | \mathscr{X}^N) = \dfrac{p(\bm{x}_N | \theta)p(\theta | \mathscr{X}^{N - 1})}{\int p(\bm{x}_N | \theta)p(\theta | \mathscr{X}^{N - 1})\, d\theta}$$其中为形式统一起见，记没有任何样本知识情况下的先验概率$p(\theta | \mathscr{X}^0) = p(\theta)$。

随着样本数的增加，可以得到一系列对概率密度函数参数的估计$$p(\theta),\quad p(\theta | \bm{x}_1),\quad p(\theta | \bm{x}_1, \bm{x}_2),\ ...,\ p(\theta | \bm{x}_1, \bm{x}_2, ..., \bm{x}_N)$$称作**递推的贝叶斯估计**。随着样本数的增加，后验概率会逐渐变得尖锐，最终趋向于以$\theta$的真实值为中心的一个尖峰，当样本无穷多时收敛于在参数真实值上的脉冲函数，则这一过程称为**贝叶斯学习**。

同时，估计的样本概率密度函数$$p(\bm{x} | \mathscr{X}^N) = \int_{\varTheta} p(\bm{x} | \theta) p(\theta | \mathscr{X}^N) \, d\theta$$也会逼近真实的密度函数：$$\lim_{N \to \infty}p(\bm{x} | \mathscr{X}^N) = p(\bm{x})$$

### 正态分布时的贝叶斯估计
假设模型均值$\mu$为待估计参数，方差$\sigma^2$已知；$\mu$的先验分布为$\mathcal{N}(\mu_0, \sigma_0^2)$。代入
$$p(\mu | \mathscr{X}) = \dfrac{p(\mathscr{X} | \mu) p(\mu)}{\int_{\varTheta}p(\mathscr{X} | \mu) p(\mu)\, d\mu}$$略过分母（其为归一项，可暂时先不考虑），得
$$p(\mathscr{X} | \mu) p(\mu) = \dfrac{1}{\sqrt{2\pi} \sigma}\exp\left(-\dfrac{1}{2}\left(\dfrac{\mu - \mu_0}{\sigma_0}\right)^2\right) \prod_{i = 1}^N \left(\dfrac{1}{\sqrt{2\pi} \sigma}\exp\left(-\dfrac{1}{2}\left(\dfrac{x_i - \mu}{\sigma}\right)^2\right)\right)$$

将所有不依赖于$\mu$的量都写入一个常数中，上式可以被整理为$$p(\mathscr{X} | \mu) p(\mu) = \alpha \exp\left( -\dfrac{1}{2}\left(\dfrac{\mu - \mu_N}{\sigma_N}\right)^2\right)$$进一步推导得$$p(\mu | \mathscr{X}) = \dfrac{1}{\sqrt{2\pi} \sigma_N} \exp\left( -\dfrac{1}{2}\left(\dfrac{\mu - \mu_N}{\sigma_N}\right)^2\right) \sim \mathcal{N}(\mu_N, \sigma_N^2)$$

其中参数$\mu_N$、$\sigma_N$满足$$\begin{cases}
    \dfrac{1}{\sigma_N^2} = \dfrac{1}{\sigma_0^2} + \dfrac{N}{\sigma^2} \\
    \\
    \mu_N = \sigma_N^2 \left(\dfrac{\mu_0}{\sigma_0^2} + \dfrac{\sum\limits_{i = 1}^N x_i}{\sigma^2}\right) \\
\end{cases} \Longrightarrow \begin{cases}
    \mu_N = \dfrac{N\sigma_0^2 m_N + \sigma^2 \mu_0}{N \sigma_0^2 + \sigma^2} \\
    \\
    \sigma_N^2 = \dfrac{\sigma_0^2 \sigma^2}{N\sigma_0^2 + \sigma^2} \\
\end{cases}$$其中$m_N = \dfrac{1}{N}\sum\limits_{i = 1}^N x_i$为所有观测样本的算术平均。

由以上贝叶斯估计，待估计的样本密度函数均值参数服从正态分布$\mathcal{N}(\mu_N, \sigma_N^2)$，则参数$\mu$的贝叶斯估计值为$$\hat{\mu} = \int \mu p(\mu | \mathscr{X}) \,d\,u = \mu_N = \dfrac{N\sigma_0^2 m_N + \sigma^2 \mu_0}{N \sigma_0^2 + \sigma^2}$$

同时可以直接求出样本的密度函数，其也服从正态分布：$$p(\bm{x} | \mathscr{X}) = \mathcal{N}(\mu_N, \sigma^2 + \sigma_N^2)$$

## 概率密度函数估计的非参数方法
### 非参数估计的基本原理与直方图方法
直方图估计做法：
1. 把样本$\bm{x}$的每个分量在其取值范围内分成$k$个等间隔的小窗。如果$\bm{x}$是$d$维向量，则这种分割，就会得到$k^d$个小体积（或称作小舱），每个小舱的体积记为$V$（对$d$维向量，$V = \prod\limits_{i = 1}^d \text{value}_i$，其中$\text{value}_i$为第$i$维上区间宽度）。
2. 统计落入每个小舱内的样本数目$q_i$。
3. 把每个小舱内的概率密度看作是常数，并用$\dfrac{q_i}{NV}$作为其估计值，其中$N$为样本总数。

### $k_N$近邻估计法
**基本做法：** 根据总样本确定一个参数$k_N$，即**在总样本数为$N$时我们要求每个小舱内拥有的样本个数**（e.g. $k_N \sim k\sqrt{N}$）。在求$\bm{x}$处的密度估计$\hat{p}(\bm{x})$时，我们调整包含$\bm{x}$的小舱的体积$V$，直到小舱内恰好落入$k_N$个样本，并用$$\hat{p}(\bm{x}) = \dfrac{k_N / N}{V} = \dfrac{k_N}{NV}$$来估算$\hat{p}(\bm{x})$。

### Parzen窗法
假设$\bm{x} \in R^d$为$d$维特征向量，并假设每个小舱是一个超立方体，它在每一维的棱长都为$h$，则小舱体积$V = h^d$；如果要计算每个小舱内落入的样本数目，可以定义如下的$d$维单位方窗函数$$\varphi((u_1, u_2, ..., u_d)^T) = \begin{cases}
    1 & |u_j| \leq \dfrac{1}{2},\ j = 1, 2, ..., d \\
    0 & \text{others}
\end{cases}$$该函数在以原点为中心的$d$维单位超正方体内取值为$1$，而其它地方取值都为$0$。从而，对于每个$\bm{x}$，要考察某个样本$\bm{x}_i$是否在这个$\bm{x}$为中心、$h$为棱长的立方小舱内，就可以通过计算$\varphi \left(\dfrac{\bm{x} - \bm{x}_i}{h}\right)$来进行。对$N$个样本$\{\bm{x}_1, \bm{x}_2, ..., \bm{x}_N\}$，落入以$\bm{x}$为中心的超立方体内的样本数就可以写成$$k_N = \sum_{i = 1}^N \varphi \left(\dfrac{\bm{x} - \bm{x}_i}{h}\right)$$

代入$\hat{p}(\bm{x}) = \dfrac{k_N / N}{V} = \dfrac{k_N}{NV}$得
$$\hat{p}(\bm{x}) = \dfrac{1}{NV}\sum_{i = 1}^N \varphi \left(\dfrac{\bm{x} - \bm{x}_i}{h}\right) = \dfrac{1}{N}\sum_{i = 1}^N \dfrac{1}{V} \varphi \left(\dfrac{\bm{x} - \bm{x}_i}{h}\right)$$

**另一种角度：** 定义**核函数**$$K(\bm{x}, \bm{x}_i) = \dfrac{1}{V}\varphi\left(\dfrac{\bm{x} - \bm{x}_i}{h}\right)$$也可记作$K(\bm{x} - \bm{x}_i)$。其反映了一个观测样本$\bm{x}_i$对在$\bm{x}$处概率密度估计的贡献。概率密度估计就是在每一点上把所有观测样本的贡献进行平均，即$$\hat{p}(\bm{x}) = \dfrac{1}{N}\sum_{i = 1}^N K(\bm{x}, \bm{x}_i)$$由于该函数需满足概率密度函数基本条件（非负且积分为$1$），从而核函数也只需满足 **非负且积分为$1$** 即可。

常见核函数有：
+ **方窗：**（得到先前所给出的结果） $$k(\bm{x}, \bm{x}_i) = \begin{cases}
    \dfrac{1}{h^d} & |x^j - x_i^j| \leq \dfrac{h}{2},\ j = 1, 2, ..., d \\
    0 & \text{others} \\
\end{cases}$$
+ **高斯窗（正态窗）：** $$\begin{aligned}
    k(\bm{x}, \bm{x}_i) =& \dfrac{1}{\sqrt{(2\pi)^d \rho^2 |\bm{Q}|}} \exp\left(-\dfrac{1}{2} \dfrac{(\bm{x} - \bm{x}_i)^T \bm{Q}^{-1} (\bm{x} - \bm{x}_i)}{\rho^2} \right) \\
    \sim& \mathcal{N}(\bm{x}_i, \rho^2 \bm{Q})
\end{aligned}$$
+ **超球窗：** $$k(\bm{x}, \bm{x}_i) = \begin{cases}
    V^{-1} & \parallel \bm{x} - \bm{x}_i \parallel \leq \rho \\
    0 & \text{others} \\
    \end{cases}$$