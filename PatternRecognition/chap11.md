# 非监督学习与聚类

## 基于模型的聚类方法
**单峰子集分离**方法：将概率分布中的每个“单峰”作为聚类的中心点。对于高维特征，将其投影到某个一维坐标上实行单峰子集分离。

具体算法步骤：
1. 主成分分析：计算所有样本$\{\bm{x}\}$的协方差矩阵并进行本征值分解，选取最大本征值对应的本征向量$\bm{u}_i$作为投影方向，将全部样本投影到该方向$v_j = \bm{u}_j^T \bm{x}$。
2. 用非参数方法估计投影后的样本的概率密度函数$p(v_j)$，例如可以用直方图法估计概率密度函数（需要根据样本数目确定适当的窗口宽度，或者尝试多种宽度，使得概率密度估计比较平滑）。
3. 用数值方法寻找$p(v_j)$中的局部极小点（密度函数的波谷），在这些极小点做垂直于$\bm{u}_j$的分类超平面，将样本分为若干个子集。如果$p(v_j)$中没有局部极小点，则选用下一个主成分作为投影方向，转步骤2。
4. 对划分出的每一个子集转第1步继续划分，直到达到预想的聚类数，或直到所得各子类样本在每个投影方向上都是单峰分布。

## 混合模型的估计
如果**已知或可以假定每个聚类中样本所服从的概率密度函数的形式**，那么总体的样本分布就是多个概率分布的和，称作**混合模型**。
可以用概率密度函数估计方法来估计参数，实现聚类划分；这一问题称为**非监督参数估计问题**。

### 混合密度的最大似然估计
#### 假设条件
+ 样本来自类别数为$c$的各类中，但不知道每个样本究竟来自哪一类。
+ 每类的先验概率$P(\omega_j),\ j = 1, 2, ..., c$已知。
+ 类条件概率密度形式$p(\bm{x} | \omega_j, \bm{\theta}_j)$已知。
+ 未知的仅是$c$个参数向量$\bm{\theta}_1, \bm{\theta}_2, ..., \bm{\theta}_c$的值。

#### 似然函数
在监督参数估计中，我们定义似然函数为样本集$\mathscr{X}$的联合密度，即$$l(\bm{\theta}) = p(\mathscr{X} | \bm{\theta})$$

该式中样本集$\mathscr{X}$为相对某一类而言的。然而，在非监督情况下，我们不知道样本所属类别，因此先定义**混合密度**：
假定样本是先按概率$P(\omega_j)$选择一个类别状态$\omega_j$，然后按类条件密度$p(\bm{x} | \omega_j, \bm{\theta}_j)$选择$\bm{x}$得到的，则这样由$c$类样本组成的混合密度，其定义为
$$p(\bm{x} | \bm{\theta}) = \sum_{j = 1}^c p(\bm{x} | \omega_j, \bm{\theta}_j)P(\omega_j)$$

其中$\bm{\theta} = (\bm{\theta}_1, \bm{\theta}_2, ..., \bm{\theta}_c)^T$。类条件密度$p(\bm{x} | \omega_j, \bm{\theta}_j)P(\omega_j)$称为**分量密度**，先验概率$P(\omega_j)$称为**混合参数**。有时候混合参数也未知，此时将其包含在未知参数中。

**非监督情况下的似然函数：** 假设有样本集$\mathscr{X} = (\bm{x}_1, \bm{x}_2, ..., \bm{x}_N)$，每个样本的类别未知，则由先前定义的混合密度概率函数，可知被观察样本的似然函数定义为
$$l(\bm{\theta}) = p(\mathscr{X} | \bm{\theta}) = \prod_{k = 1}^N p(\bm{x}_k | \bm{\theta})$$

**对数似然函数**定义为
$$H(\bm{\theta}) = \ln(l(\bm{\theta})) = \sum_{k = 1}^N \ln p(\bm{x}_k | \bm{\theta})$$

最大似然估计就是取$\hat{\bm{\theta}} \in \varTheta$使得$l(\bm{\theta})$或$H(\bm{\theta})$最大。

#### 可识别性问题
基本目的：利用这个混合密度中抽取的样本来估计未知参数向量$\bm{\theta}$。一旦求出估计量$\hat{\bm{\theta}}$，便可以将其分解为$\hat{\bm{\theta}}_1, \hat{\bm{\theta}}_2, ..., \hat{\bm{\theta}}_c$。
如果能产生混合密度$p(\bm{x} | \bm{\theta})$的$\bm{\theta}$只有一个，那么原则上存在唯一解；然而如果存在多个符合条件的$\bm{\theta}$，则无法获得唯一解。因此定义**可识别性**：对于$\bm{\theta} \neq \bm{\theta}'$，混合分布中总存在$\bm{x}$，使得$p(\bm{x} | \bm{\theta}) \neq p(\bm{x} | \bm{\theta}')$。

通常，**大部分常见连续随机变量的分布密度函数都是可识别的，而离散随机变量的混合函数则往往是不可识别的**。

#### 计算问题
##### $P(w_i)$已知
假定似然函数$p(\bm{x} | \bm{\theta})$对$\bm{\theta}$可微，利用对数似然函数$H(\bm{\theta})$对$\bm{\theta}_i,\ i = 1, 2, ..., c$分别求导
$$\begin{aligned}
\nabla_{\bm{\theta}_i} H(\bm{\theta}) =& \nabla_{\bm{\theta}_i} \left( \sum_{k = 1}^N \ln p(\bm{x}_k | \bm{\theta}) \right) \\
=& \sum_{k = 1}^N \nabla_{\bm{\theta}_i} \ln p(\bm{x}_k | \bm{\theta}) \\
=& \sum_{k = 1}^N \dfrac{1}{p(\bm{x}_k | \bm{\theta})} \nabla_{\bm{\theta}_i} p(\bm{x}_k | \bm{\theta}) \\
=& \sum_{k = 1}^N \dfrac{1}{p(\bm{x}_k | \bm{\theta})} \nabla_{\bm{\theta}_i} \left(\sum_{j = 1}^c p(\bm{x}_k | \bm{\theta}_j) \right) \\
=& \sum_{k = 1}^N \dfrac{1}{p(\bm{x}_k | \bm{\theta})} \nabla_{\bm{\theta}_i} \left(\sum_{j = 1}^c p(\bm{x}_k | \omega_j, \bm{\theta}_j)P(\omega_j) \right)
\end{aligned}$$

当$i \neq j$时$\bm{\theta}_i$和$\bm{\theta}_j$的元素在函数上是独立的，从而再对原式进行变形：
$$\begin{aligned}
\nabla_{\bm{\theta}_i} H(\bm{\theta}) =& \sum_{k = 1}^N \dfrac{1}{p(\bm{x}_k | \bm{\theta})} \nabla_{\bm{\theta}_i} \left(\sum_{j = 1}^c p(\bm{x}_k | \omega_j, \bm{\theta}_j)P(\omega_j) \right) \\
=& \sum_{k = 1}^N \dfrac{1}{p(\bm{x}_k | \bm{\theta})} \nabla_{\bm{\theta}_i} \left(p(\bm{x}_k | \omega_i, \bm{\theta}_i)P(\omega_i) \right) \\
=& \sum_{k = 1}^N \dfrac{P(\omega_i)}{p(\bm{x}_k | \bm{\theta})} \nabla_{\bm{\theta}_i} p(\bm{x}_k | \omega_i, \bm{\theta}_i)
\end{aligned}$$

并且引进后验概率
$$p(\omega_i | \bm{x}_k, \bm{\theta}_i) = \dfrac{p(\bm{x}_k | \omega_i, \bm{\theta}_i)P(\omega_i)}{p(\bm{x}_k | \bm{\theta})}$$

即
$$\dfrac{P(\omega_i)}{p(\bm{x}_k | \bm{\theta})} = \dfrac{p(\omega_i | \bm{x}_k, \bm{\theta}_i)}{p(\bm{x}_k | \omega_i, \bm{\theta}_i)}$$

从而再进行变形：
$$\begin{aligned}
\nabla_{\bm{\theta}_i} H(\bm{\theta}) =& \sum_{k = 1}^N \dfrac{P(\omega_i)}{p(\bm{x}_k | \bm{\theta})} \nabla_{\bm{\theta}_i} p(\bm{x}_k | \omega_i, \bm{\theta}_i) \\
=& \sum_{k = 1}^N \dfrac{p(\omega_i | \bm{x}_k, \bm{\theta}_i)}{p(\bm{x}_k | \omega_i, \bm{\theta}_i)} \nabla_{\bm{\theta}_i} p(\bm{x}_k | \omega_i, \bm{\theta}_i) \\
=& \sum_{k = 1}^N p(\omega_i | \bm{x}_k, \bm{\theta}_i) \left( \dfrac{1}{p(\bm{x}_k | \omega_i, \bm{\theta}_i)} \nabla_{\bm{\theta}_i} p(\bm{x}_k | \omega_i, \bm{\theta}_i) \right) \\
=& \sum_{k = 1}^N p(\omega_i | \bm{x}_k, \bm{\theta}_i) \nabla_{\bm{\theta}_i} \ln p(\bm{x}_k | \omega_i, \bm{\theta}_i) \\
\end{aligned}$$

令上式等于$0$，得到最大似然估计必须满足的条件
$$\sum_{k = 1}^N p(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) \nabla_{\bm{\theta}_i} \ln p(\bm{x}_k | \omega_i, \hat{\bm{\theta}}_i) = 0,\quad i = 1, 2, ..., c$$

上式实际为$c$个微分方程组组成的方程组，解为$\bm{\theta}$的最大似然估计$\hat{\bm{\theta}} = (\hat{\bm{\theta}}_1, \hat{\bm{\theta}}_2, ..., \hat{\bm{\theta}}_c)^T$

##### $P(w_i)$未知
如果未知量中包含$P(\omega_i)$，则最大似然值的搜索需要添加额外的限制条件
$$P(\omega_i) \geq 0,\quad i = 1, 2, ..., c \\ \sum_{i = 1}^c P(\omega_i) = 1$$

设$\hat{P}(\omega_i)$为$P(\omega_i)$的最大似然估计，$\hat{\bm{\theta}}_i$为$\bm{\theta}_i$的最大似然估计。

$$H = \sum_{k = 1}^N \ln p(\bm{x}_k | \bm{\theta}) = \sum_{k = 1}^N \ln \left( \sum_{i = 1}^c p(\bm{x}_k | \omega_i, \bm{\theta}_i)P(\omega_i) \right)$$

从而可以写出拉格朗日函数
$$\begin{aligned}
    H' =& H + \lambda \left( \sum_{i = 1}^c P(\omega_i) - 1 \right) \\
    =& \sum_{k = 1}^N \ln \left( \sum_{i = 1}^c p(\bm{x}_k | \omega_i, \bm{\theta}_i)P(\omega_i) \right) + \lambda \left( \sum_{i = 1}^c P(\omega_i) - 1 \right)
\end{aligned}$$

对$P(\omega_i)$求导，得
$$\dfrac{\partial H'}{\partial P(\omega_i)} = \sum_{k = 1}^N \dfrac{p(\bm{x}_k | \omega_i, \bm{\theta}_i)}{\sum\limits_{j = 1}^c p(\bm{x}_k | \omega_j, \bm{\theta}_j) \hat{P}(\omega_j)} + \lambda = 0,\quad i = 1, 2, ..., c$$

由贝叶斯公式
$$\begin{aligned}
    \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) =& \dfrac{p(\bm{x}_k | \omega_i, \bm{\theta}_i)\hat{P}(\omega_i)}{p(\bm{x}_k | \bm{\theta}_i)} \\
    =& \dfrac{p(\bm{x}_k | \omega_i, \bm{\theta}_i)\hat{P}(\omega_i)}{\sum\limits_{j = 1}^c p(\bm{x}_k | \omega_j, \bm{\theta}_j) \hat{P}(\omega_j)}
\end{aligned}$$

即
$$\begin{aligned}
    \dfrac{\hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i)}{\hat{P}(\omega_i)} = \dfrac{p(\bm{x}_k | \omega_i, \bm{\theta}_i)}{\sum\limits_{j = 1}^c p(\bm{x}_k | \omega_j, \bm{\theta}_j) \hat{P}(\omega_j)}
\end{aligned}$$

从而上式可转写为
$$\begin{aligned}
    \dfrac{\partial H'}{\partial P(\omega_i)} =& \sum_{k = 1}^N \dfrac{p(\bm{x}_k | \omega_i, \bm{\theta}_i)}{\sum\limits_{j = 1}^c p(\bm{x}_k | \omega_j, \bm{\theta}_j) \hat{P}(\omega_j)} + \lambda \\
    =& \sum_{k = 1}^N \dfrac{\hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i)}{\hat{P}(\omega_i)} + \lambda= 0
\end{aligned}$$

即
$$\sum_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) = - \lambda \hat{P}(\omega_i),\quad i = 1, 2, ..., c$$

将以上$c$个方程相加，有
$$N = \sum_{i = 1}^c \sum_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) = -\lambda \sum_{i = 1}^c \hat{P}(\omega_i) = -\lambda$$

将结果代入回$\sum\limits_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) = - \lambda \hat{P}(\omega_i)$，得到$P(\omega_i)$的最大似然估计$\hat{P}(\omega_i)$：
$$\hat{P}(\omega_i) = \dfrac{1}{N} \sum_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i),\quad i = 1, 2, ..., c$$

而
$$\hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) = \dfrac{p(\bm{x}_k | \omega_i, \hat{\bm{\theta}}_i) \hat{P}(\omega_i)}{\sum\limits_{j = 1}^c p(\bm{x}_k | \omega_j, \hat{\bm{\theta}}_j) \hat{P}(\omega_j)},\quad i = 1, 2, ..., c$$

理论上，$\hat{\bm{\theta}}$和$\hat{P}(\omega_j)$可以从以上微分方程组中解出，但实际直接求出闭式解过于复杂，通常只能采用迭代法求解。

### 混合正态分布的参数估计
假设各个分布都是多维正态分布，即$p(\bm{x} | \omega_i, \bm{\theta}_i) \sim \mathcal{N}(\bm{\mu}_i, \bm{\Sigma}_i)$，总共有$c$个概率分布，各个概率分布的先验概率为$P(\omega_i),\ i = 1, 2, ..., c$。这是最常用的混合模型，称作**混合高斯模型**。

#### 情况一：$\bm{\mu}_i$未知，$\bm{\Sigma}_i$、$P(\omega_i)$、$c$已知
此时参数$\bm{\theta}_i$就是$\bm{\mu}_i$。
注意到
$$\begin{gathered}
    \ln p(\bm{x} | \omega_i, \hat{\bm{\mu}}_i) = -\ln((2\pi)^{\frac{d}{2}} |\bm{\Sigma}_i|^{\frac{1}{2}}) - \dfrac{1}{2}(\bm{x} - \bm{\mu}_i)^T \bm{\Sigma}_i^{-1} (\bm{x} - \bm{\mu}_i) \\
    \nabla_{\bm{\mu}_i} \ln p(\bm{x} | \omega_i, \hat{\bm{\mu}}_i) = \bm{\Sigma}_i^{-1} (\bm{x} - \bm{\mu}_i)
\end{gathered}$$

代入$\sum\limits_{k = 1}^N p(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) \nabla_{\bm{\theta}_i} \ln p(\bm{x}_k | \omega_i, \hat{\bm{\theta}}_i) = 0$可得

$$\sum\limits_{k = 1}^N p(\omega_i | \bm{x}_k, \hat{\bm{\mu}}_i) \bm{\Sigma}_i^{-1} (\bm{x}_k - \bm{\mu}_i) = 0$$

两边同乘$\bm{\Sigma}_i$并移项，得到

$$\hat{\bm{\mu}}_i = \dfrac{\sum\limits_{k = 1}^N P(\omega_i | \bm{x}_k, \hat{\bm{\mu}}_i) \bm{x}_k}{\sum\limits_{k = 1}^N P(\omega_i | \bm{x}_k, \hat{\bm{\mu}}_i)},\quad i = 1, 2, ..., c$$

这一方程标明：$\bm{\mu}_i$的最大似然估计就是样本的加权平均，其中样本$\bm{x}_k$的权值为属于第$i$类有多大可能性的估计。例：$P(\omega_i | \bm{x}_k, \hat{\bm{\mu}}_i)$对部分样本为$1$，对剩余样本为$0$，则$\bm{\mu}_i$被估计为属于第$i$类的那些样本的平均。
然而问题在于，$P(\omega_i | \bm{x}_k, \hat{\bm{\mu}}_i)$是未知的。使用贝叶斯公式
$$P(\omega_i | \bm{x}_k, \hat{\bm{\mu}}_i) = \dfrac{p(\bm{x}_k | \omega_i, \hat{\bm{\mu}}_i) P(\omega_i)}{\sum\limits_{j = 1}^c p(\bm{x}_k | \omega_j, \hat{\bm{\mu}}_j) P(\omega_j)}$$

并将$p(\bm{x} | \omega_i, \bm{\theta}_i) \sim \mathcal{N}(\bm{\mu}_i, \bm{\Sigma}_i)$代入，$\hat{\bm{\mu}}_i$的表达式将构成一组十分复杂的非线性联立方程组，不仅求解困难，而且一般没有唯一解。

我们通常使用**迭代法**，在设定初始估计$\bm{\mu}_i$后得到迭代算法：
$$\hat{\bm{\mu}}_i(t + 1) = \dfrac{\sum\limits_{k = 1}^N P(\omega_i | \bm{x}_k, \hat{\bm{\mu}}_i(t)) \bm{x}_k}{\sum\limits_{k = 1}^N P(\omega_i | \bm{x}_k, \hat{\bm{\mu}}_i(t))},\quad i = 1, 2, ..., c$$

注意这种算法得到的不是全局最优解而是局部最优解，甚至可能收敛到鞍点，因此需要对运算结果进行更多分析。

#### 情况二：$\bm{\mu}_i$、$\bm{\Sigma}_i$、$P(\omega_i)$未知，$c$已知
用$\bm{\theta}$代表$\bm{\mu}_i$、$\bm{\Sigma}_i$、$P(\omega_i)$等未知参数。注意到限制条件$P(\omega_i) \geq 0,\ i = 1, 2, ..., c$和$\sum\limits_{i = 1}^c P(\omega_i) = 1$仍然存在，因此同样可用拉格朗日乘子法对对数似然函数$H(\bm{\theta})$构造拉格朗日函数$H' = H + \lambda\left(\sum\limits_{i = 1}^c P(\omega_i) - 1\right)$，让$H'$分别对$\bm{\mu}_i$、$\bm{\Sigma}_i$和$P(\omega_i)$求导并令其等于零，解得
$$\begin{aligned}
    \hat{P}(\omega_i) =& \dfrac{1}{N} \sum_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) \\
    \hat{\bm{\mu}}_i =& \dfrac{\sum\limits_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) \bm{x}_k}{\sum\limits_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i)} \\
    \hat{\bm{\Sigma}}_i =& \dfrac{\sum\limits_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) (\bm{x}_k - \hat{\bm{\mu}}_i) (\bm{x}_k - \hat{\bm{\mu}}_i)^T}{\sum\limits_{k = 1}^N \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i)}
\end{aligned}$$

其中
$$\begin{aligned}
    \hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) =& \dfrac{P(\bm{x}_k | \omega_i, \hat{\bm{\theta}}_i) \hat{P}(\omega_i)}{\sum\limits_{j = 1}^c P(\bm{x}_k | \omega_i, \hat{\bm{\theta}}_i) \hat{P}(\omega_j)} \\
    =& \dfrac{|\hat{\bm{\Sigma}}_i|^{-\frac{1}{2}} \exp\left(-\dfrac{1}{2} (\bm{x}_k - \hat{\bm{\mu}}_i)^T \hat{\bm{\Sigma}}_i^{-1} (\bm{x}_k - \hat{\bm{\mu}}_i)\right) \hat{P}(\omega_i)}{\sum\limits_{j = 1}^c |\hat{\bm{\Sigma}}_j|^{-\frac{1}{2}} \exp\left(-\dfrac{1}{2} (\bm{x}_k - \hat{\bm{\mu}}_j)^T \hat{\bm{\Sigma}}_j^{-1} (\bm{x}_k - \hat{\bm{\mu}}_j)\right) \hat{P}(\omega_j)}
\end{aligned}$$

在极端情况下，即当$\bm{x}_k$来自$\omega_i$类时，$\hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) = 1$，否则$\hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i) = 0$，此时三个估计变为
$$\begin{aligned}
    \hat{P}(\omega_i) =& \dfrac{N_i}{N} \\
    \hat{\bm{\mu}}_i =& \dfrac{1}{N_i}\sum_{k = 1}^{N_i} \bm{x}_k^{(i)} \\
    \hat{\bm{\Sigma}}_i =& \dfrac{1}{N_i}\sum_{k = 1}^{N_i} (\bm{x}_k^{(i)} - \hat{\bm{\mu}}_i)^T (\bm{x}_k^{(i)} - \hat{\bm{\mu}}_i)
\end{aligned}$$

其中$N_i$为$\omega_i$类的样本数，$\bm{x}_k^{(i)}$为来自$\omega_i$类的样本。

事实上，求解这些方程是相对困难的，有效的方法仍然是采用迭代法，即用一个初始值计算$\hat{P}(\omega_i | \bm{x}_k, \hat{\bm{\theta}}_i)$，然后用$\hat{\bm{\mu}}_i$、$\hat{\bm{\Sigma}}_i$、$\hat{P}(\omega_i)$的表达式反复迭代。

## 动态聚类算法
### $C$均值算法（$K$均值算法）
$C$均值算法基于最小误差平方和准则。
若$N_i$为第$i$聚类$\varGamma_i$中的样本数目，$\bm{m}_i$是这些样本的均值，即$\bm{m}_i = \dfrac{1}{N_i} \sum\limits_{\bm{y} \in \varGamma_i} \bm{y}$。把$\varGamma_i$中的各个样本$\bm{y}$与均值$\bm{m}_i$间的误差平方和对所有类相加后得到
$$J_e = \sum_{i = 1}^c \sum_{\bm{y} \in \varGamma_i} \parallel \bm{y} - \bm{m}_i \parallel^2$$

这里$J_e$就是**误差平方和聚类准则**，它是样本集$\mathscr{Y}$和类别集$\varOmega$的函数。$J_e$度量了用$c$个聚类中心$\bm{m}_i,\ i = 1, 2, ..., c$代表$c$个样本子集$\varGamma_i,\ i = 1, 2, ..., c$时，所产生的总的误差平方。我们的聚类希望最小化$J_e$。

这种误差平方和无法用解析的方法最小化，只能通过迭代的方法，通过不断调整样本的类别归属来求解。
假设已经有一个划分方法，它把样本$\bm{y}$划分在类别$\varGamma_k$中。考查下面的调整：如果将$\bm{y}$从$\varGamma_k$移动到$\varGamma_j$类中，则这两个类别发生了变化，$\varGamma_k$少了一个样本而变成$\tilde{\varGamma_k}$，$\varGamma_j$类多了一个样本从而变成$\tilde{\varGamma_j}$，其余类别不受影响。调整后，两类均值变为
$$\begin{aligned}
    \tilde{\bm{m}}_k =& \bm{m}_k + \dfrac{\bm{m}_k - \bm{y}}{N_k - 1} \\
    \tilde{\bm{m}}_j =& \bm{m}_j + \dfrac{\bm{m}_j - \bm{y}}{N_j + 1} \\
\end{aligned}$$

两类误差平方和也分别变为
$$\begin{aligned}
    \tilde{J}_k =& J_k - \dfrac{N_k}{N_k - 1} \parallel \bm{y} - \bm{m}_k \parallel^2 \\
    \tilde{J}_j =& J_j + \dfrac{N_j}{N_j + 1} \parallel \bm{y} - \bm{m}_j \parallel^2 \\
\end{aligned}$$

总的误差平方和只取决于这两个变化。
事实上，只要移动带来的$\varGamma_k$类均方误差的减小量大于$\varGamma_j$类均方误差的增加量，即
$$\dfrac{N_j}{N_j + 1}\parallel \bm{y} - \bm{m}_j \parallel^2 < \dfrac{N_k}{N_k - 1}\parallel \bm{y} - \bm{m}_k \parallel^2$$

则这一步搬运就有利于总体误差平方和的减少。
如果类别总数$c > 2$，则可考查$\varGamma_k$类之外的所有其他类，选择其中均方误差增加量最小的类别，如果其小于$\varGamma_k$类均方误差的减小量，则将$\bm{y}$从$\varGamma_k$类移动到该新类别中。

**具体步骤：**
1. 初始划分$C$个聚类$\varGamma_i,\ i = 1, 2, ..., c$，计算$\bm{m}_i,\ i = 1, 2, ..., c$和$J_e$。
2. 任取一个样本$\bm{y}$，设$\bm{y} \in \varGamma_i$。
3. 若$N_i = 1$，则转第2步，否则继续。
4. 计算$$\begin{aligned}
    \rho_i =& \dfrac{N_i}{N_i - 1} \parallel \bm{y} - \bm{m}_i \parallel^2 \\
    \rho_j =& \dfrac{N_j}{N_j + 1} \parallel \bm{y} - \bm{m}_j \parallel^2,\quad j \neq i \\
\end{aligned}$$
5. 取$\rho_k = \min\limits_{j \neq i} \rho_j$，若$\rho_k < \rho_i$，则把$\bm{y}$从$\varGamma_i$类移动到$\varGamma_k$类中。
6. 重新计算$\bm{m}_i,\ i = 1, 2, ..., c$和$J_e$。
7. 如果连续$N$次迭代后$J_e$不改变，则停止，否则转步骤2。

#### 初始划分
常用方法：
1. 凭经验选择代表点。
2. 将全部数据随机地分成$c$类，计算每类重心，将这些重心作为每类的代表点。
3. 用“密度”法选择代表点。这里“密度”是具有统计性质的样本密度。
   1. 一种方法为：以每个样本为球心，用某个正数$\xi$为半径作一个球形区域，落在该球内的样本数则称为该点的“密度”。计算全部点的“密度”后。
   2. 首先选择“密度”最大的样本点作为第一个代表点。
   3. 之后在选择第二个代表点时，人为规定一个数值$\delta > 0$，要求在离开第一个代表点$\delta$距离以外选择次大“密度”点作为第二个代表点，这样就可以避免代表点可能集中在一起的问题。
   4. 之后的代表点依此类推。
4. 按照样本天然的排列顺序，或者将样本随机排序后，用前$c$个点作为代表点。
5. 从$(c - 1)$聚类划分问题中产生$c$聚类划分问题的代表点。
   1. 先将全部样本看做一个聚类，其代表点为样本总均值。
   2. 确定二聚类问题的代表点，分别为一聚类划分的总均值和距其最远的点。
   3. 依此类推，$c$聚类划分问题的代表点就是$(c - 1)$聚类划分最后得到的各均值再加上离最近的均值最远的点。

根据不同的数据分布和先验知识选择不同的方法；当对数据的知识不足时，可以采用方法2或方法4的随机初始代表点方法。必要时可以用不同的初始代表点进行多次$C$均值聚类，再对结果进行选择或融合。

#### 初始分类
常见方法：
1. 选择一批代表点后，将剩余点归类至最近的代表点所在类。
2. 选择一批代表点后，每个代表点自成一类，将样本依顺序归入与其最近的代表点的那一类，并立即重新计算该类的重心以代替原来的代表点。之后计算下一个样本的归类，直到所有的样本都归到相应的类中。
3. 规定一个整数$\xi$，选择$\omega_1 = \{\bm{y}_1\}$，计算样本$\bm{y}_2$与$\bm{y}_1$间的距离$\delta(\bm{y}_2, \bm{y}_1)$，如果小于$\xi$则将$\bm{y}_2$归入$\omega_1$，否则建立新类$\bm{y}_2$。当某一步轮到$\bm{y}_l$归入时，假设当时已经形成了$k$类$\omega_1, \omega_2, ..., \omega_k$，而每个类第一个归入的样本记作$\bm{y}_1^1, \bm{y}_2^1, ..., \bm{y}_k^1$。若$\delta(\bm{y}, \bm{y}_i^1) > \xi,\ i = 1, 2, ..., k$则将$\bm{y}_l$建立为第$k + 1$类，即$\omega_k = \{\bm{y}_l\}$，否则将$\bm{y}_l$归入与$\bm{y}_1^1, \bm{y}_2^1, ..., \bm{y}_k^1$距离最近的那一类。
4. 先将数据标准化，用$y_{ij}$表示标准化后第$i$个样本的第$j$个坐标。令$$\begin{gathered}
    \text{SUM}(i) = \sum_{j = 1}^d y_{ij} \\
    \text{MA} = \max_i \text{SUM}(i) \\
    \text{MI} = \min_i \text{SUM}(i) \\
\end{gathered}$$如果希望将样本划分为$c$类，则对每个$i$计算$$\dfrac{(c - 1)(\text{SUM}(i) - \text{MI})}{\text{MA} - \text{MI}} + 1$$假设与这个计算值最接近的整数为$k$，则将$\bm{y}_i$归入第$k$类。

#### 关于$C$均值方法中的聚类数目$c$
$C$均值方法的前提是$c$是事先给定的，然而这在某些非监督学习中不一定总是能被满足。
当类别数目未知的情况下，可以逐一用$c = 1, c = 2, ...$来进行聚类，每一次聚类都计算出最后达到的误差平方和$J_e(c)$，通过考查$J_e(c)$随$c$的变化而推断合理的类别数。
一种常用方法是绘制$J_e(c) - c$图，选择较为明显的拐点对应的$c$作为最终聚类的类别数目。但并非所有情况下都能找到明显的拐点，此时只能根据先验知识或不同类别数下的结果比较来确定最合理的聚类数。

### ISODATA方法
与$C$均值算法有两点不同：
+ 将全部样本调整完后，才计算各类均值；
+ 聚类过程中引入对类别的评判准则，可以根据这些准则自动地将某些类别合并或分裂，从而使得聚类结果更加合理。

**具体步骤：**
设由$N$个样本组成的样本集为$\{\bm{y}_1, \bm{y}_2, ..., \bm{y}_M\} \subset \mathbb{R}^d$，事先确定以下参数：
期望得到的聚类数$K$，每个聚类中最少样本数$\theta_N$，标准偏差参数$\theta_S$，合并参数$\theta_C$，每次迭代允许合并的最大聚类对数$L$，允许迭代的次数$I$。
1. 初始化，设初始聚类数$c$（不一定等于期望聚类数$K$），用与$C$均值聚类法相同的方式确定$c$个初始中心$\bm{m}_i,\ i = 1, 2, ..., c$。
2. 把所有样本分到距离中心最近的类$\varGamma_i$中，$i = 1, 2, ..., c$。
3. 若某个类$\varGamma_j$中样本数过少（$N_j < \theta_N$），则去掉这一类，其中的样本根据其他类中心的距离分入其他类，置$c = c - 1$。
4. 重新计算均值$$\bm{m}_j = \dfrac{1}{N_j} \sum_{\bm{y} \in \varGamma_j} \bm{y},\quad j = 1, 2, ..., c$$其中$N_j$为第$j$个聚类中的样本数目（基数）。
5. 计算第$j$类样本与其中心的平均距离$$\bar{\delta}_j = \dfrac{1}{N_j} \sum_{\bm{y} \in \varGamma_j} \parallel \bm{y} - \bm{m}_j \parallel,\quad j = 1, 2, ..., c$$和总平均距离$$\bar{\delta} = \dfrac{1}{N} \sum_{j = 1}^c N_j \bar{\delta}_j$$
6. 如果这是最后一次迭代（由参数$I$决定），则程序停止；否则，
   + 若$c \leq \dfrac{K}{2}$，则转步骤7（分裂）。
   + 若$c \geq 2K$，或是偶数次迭代，则转步骤8（合并）。
7. 分裂：
   1. 对每个类，用下面的公式求各维标准差$\bm{\sigma}_j = (\sigma_{j1}, \sigma_{j2}, ..., \sigma_{jd})^T$：$$\sigma_{ji} = \sqrt{\dfrac{1}{N_j} \sum_{y_{ki} \in \varGamma_j}(y_{ki} - m_{ji})^2},\quad j = 1, 2, ..., c,\quad i = 1, 2, ..., d$$其中，$y_{ki}$为第$k$个样本的第$i$个分量，$m_{ji}$是当前第$j$个聚类中心的第$i$个分量，$\sigma_{ji}$是第$j$类第$i$个分量的标准差，$d$为样本维数。
   2. 对每个类，求出标准偏差最大的分量$\sigma_{j \max},\ j = 1, 2, ..., c$。
   3. 对各类的$\sigma_{j \max},\ j = 1, 2, ..., c$，若存在某个类$\sigma_{j \max} > \theta_s$（标准偏差参数），且该类样本与其中心的平均距离大于总平均距离（即$\bar{\delta}_j > \bar{\delta}$），且该类样本数$N_j > 2(\theta_N + 1)$，则将$\varGamma_j$分裂为两类，中心分别为$\bm{m}_j^+$和$\bm{m}_j^-$：$$\bm{m}_j^+ = \bm{m}_j + \bm{\gamma}_j \qquad \bm{m}_j^- = \bm{m}_j - \bm{\gamma}_j$$其中，分裂项$\bm{\gamma}_j$可以为$\bm{\gamma}_j = k\bm{\sigma}_j$；也可以为$\bm{\gamma}_j = (0, 0, ..., 0, \sigma_{j \max}, 0, ..., 0)^T$，即只在$\sigma_{j \max}$对应的特征分量上将这一类分裂开。置$c = c + 1$。
8. 合并：
   1. 计算各类中心两两之间的距离$$\delta_{ij} = \parallel \bm{m}_i - \bm{m}_j \parallel,\quad i, j = 1, 2, ..., c,\quad i \neq j$$
   2. 比较$\delta_{ij}$与$\theta_c$（合并参数），对小于$\theta_c$者排序$$\delta_{(i_1, j_1)} < \delta_{(i_2, j_2)} < ... < \delta_{(i_l, j_l)}$$
   3. 从最小的$\delta_{(i_1, j_1)}$开始，把每个$\delta_{(i_k, j_k)}$对应的$\bm{m}_{i_k}$和$\bm{m}_{j_k}$合并，组成新类，新的中心为$$\bm{m}_k = \dfrac{1}{N_{i_k} + N_{j_k}}(N_{i_k} \bm{m}_{i_k} + N_{j_k} \bm{m}_{j_k})$$
9.  如果这是最后一次迭代，则程序停止；否则，迭代次数$+ 1$，转步骤2。

### 基于核的动态聚类算法
可以定义一个核$K_j = K(\bm{y}, V_j)$来代表一个类$\varGamma_j$，其中$V_j$表示参数集。核$K$可以是一个函数、一个点集或者其他能表示类别的模型；此外，我们还需要定义一个样本$\bm{y}$到核的距离$\varDelta(\bm{y}, K_j)$。
类似$C$均值算法，定义准则函数
$$J_K = \sum_{j = 1}^c \sum_{\bm{y} \in \varGamma_j} \varDelta(\bm{y}, K_j)$$

当$\varDelta$在此处表示某种距离度量时，我们希望算法使得$J_K$最小。
**具体步骤：**
1. 选择初始划分，即将样本集$\mathscr{Y}$划分成$c$类，并确定每类的初始核$K_j,\ j = 1, 2, ..., c$。
2. 若$\varDelta(\bm{y}, K_j) = \min\limits_k \varDelta(\bm{y}, K_k),\ k = 1, 2, ..., c$，则$\bm{y} \in \varGamma_j$，即将样本$\bm{y}$归到$\varGamma_j$类中。
3. 重新修正核$K_j, j = 1, 2, ..., c$。若核$K_j$保持不变，则算法中止；否则转步骤2。

可以看到，$C$均值算法实际上是这一算法的一个特例，其中核为类均值，而样本到均值的欧式距离为度量。 
此外还有一些其他的核类型。

#### 正态核函数
如果样本分布为椭圆状正态分布，则可以用正态核函数来代表类：
$$K_j(\bm{y}, V_j) = \dfrac{1}{(2 \pi)^{\frac{d}{2}} |\hat{\bm{\Sigma}}_j|^{\frac{1}{2}}}\exp\left( -\dfrac{1}{2}(\bm{y} - \bm{m}_j)^T \hat{\bm{\Sigma}}_j^{-1} (\bm{y} - \bm{m}_j) \right)$$

其中参数集为$V_j = \{\bm{m}_j, \hat{\bm{\Sigma}}_j\}$，$\bm{m}_j$为第$j$个类的样本均值，$\hat{\bm{\Sigma}}_j$为该类的样本协方差矩阵。
我们定义样本到核的相似性度量为核函数的负对数去掉常数项，即
$$\varDelta(\bm{y}, K_j) = -\log K_j(\bm{y}, V_j) - C = \dfrac{1}{2}(\bm{y} - \bm{m}_j)^T \hat{\bm{\Sigma}}_j^{-1} (\bm{y} - \bm{m}_j) + \dfrac{1}{2}\log |\hat{\bm{\Sigma}}_j|$$

#### 主轴核函数
有些情况下，样本集中在相应主轴方向的子空间中；我们可以用K-L变换得到样本的主轴，从而可以定义核函数
$$K(\bm{y}, V_j) = \bm{U}_j^T \bm{y}$$

其中$\bm{U}_j = (\bm{u}_1^T, \bm{u}_2^T, ..., \bm{u}_{d_j}^T)^T$是和第$j$个类的样本协方差矩阵$\hat{\bm{\Sigma}}_j$的前$d_j$个最大本征值相对应的本征向量**系统**。
任何一个样本$\bm{y}$和$\varGamma_i$之间的相似性程度可以用$\bm{y}$与$\varGamma$类主轴之间的欧氏距离的平方来度量：
$$\begin{aligned}
    \varDelta(\bm{y}, K_j) =& ((\bm{y} - \bm{m}_j) - \bm{U}_j \bm{U}_j^T (\bm{y} - \bm{m}_j))^T \\
    & ((\bm{y} - \bm{m}_j) - \bm{U}_j \bm{U}_j^T (\bm{y} - \bm{m}_j))
\end{aligned}$$

## 模糊聚类方法
### 模糊集的基本知识
**隶属度函数：** 表示一个对象$\bm{x}$隶属于集合$A$程度的函数，通常记作$\mu_A(\bm{x})$。其自变量范围为所有可能属于集合$A$的对象，取值范围为$[0, 1]$，$\mu_A(\bm{x}) = 1$表示$\bm{x}$完全属于$A$，相当于传统集合概念上的$\bm{x} \in A$；而$\mu_A(\bm{x}) = 0$表示$\bm{x}$完全不属于$A$，相当于传统集合概念上的$\bm{x} \notin A$。

一个定义在空间$X = \{\bm{x}\}$上的隶属度函数就定义了一个**模糊集合**$A$，或者叫做定义在空间$X = \{\bm{x}\}$上的一个**模糊子集**$A$。对于有限个对象$\bm{x}_1, \bm{x}_2, ..., \bm{x}_n$，模糊集合$A$可以表示为$$A = \{(\mu_A (\bm{x}_i), \bm{x}_i)\} \quad\text{or}\quad A = \bigcup_i \dfrac{\mu_i}{\bm{x}_i}$$

此处$\bm{x}_i$仍可叫做$A$中的元素。与模糊集合相对应，传统的集合可以叫做**确定集合**或**脆集合**，下文中未明确说明的集合即指确定集合。显然，确定集是模糊集的特例，即隶属度函数只取$1$或$0$。
空间$X$中$A$的隶属度大于$0$的对象的集合叫做模糊集$A$的**支持集**$S(A)$，即$S(A) = \{\bm{x},\ \bm{x} \in X,\ \mu_A(\bm{x}) > 0\}$。支持集中的元素称为模糊集$A$的**支持点**，或不严格地称为模糊集$A$的元素。

### 模糊$C$均值算法
符号重新规定：
$\{\bm{x}_i,\ i = 1, 2, ..., n\}$为$n$个样本组成的样本集合，$c$为预定的类别数目，$\bm{m}_i,\ i = 1, 2, ..., c$为每个聚类的中心，$\mu_j(\bm{x}_i)$为第$i$个样本对于第$j$类的隶属度函数。用隶属度函数定义的聚类损失函数为
$$J_f = \sum_{j = 1}^c \sum_{i = 1}^{n} (\mu_j(\bm{x}_i))^b \parallel \bm{x}_i - \bm{m}_j \parallel^2$$

其中$b > 1$为一个可以控制聚类结果模糊程度的常数，如果$b \to 1$则算法将得到等同于$C$均值算法的确定性聚类划分；如果$b = \infty$，则算法将得到完全模糊的解，即各类中心都收敛到所有训练样本的中心，同时所有样本都以等同的概率归属各个类，因而完全失去分类意义。通常，人们取$b = 2$。

模糊$C$均值算法规定**一个样本对于各个聚类的隶属度之和为$1$** ，即
$$\sum_{j = 1}^c \mu_j(\bm{x}_i) = 1,\quad i = 1, 2, ..., n$$

在这一约束下求$J_f$最小值，令$J_f$对$\bm{m}_j$和$\mu_j(\bm{x}_i)$的偏导数为$0$，可得必要条件
$$\begin{gathered}
    \bm{m}_j = \dfrac{\sum\limits_{i = 1}^n (\mu_j(\bm{x}_i))^b \bm{x}_i}{\sum\limits_{i = 1}^n (\mu_j(\bm{x}_i))^b},\quad j = 1, 2, ..., c \\
    \mu_j(\bm{x}_i) = \dfrac{\left( \dfrac{1}{\parallel \bm{x}_i - \bm{m}_j \parallel^2} \right)^{\frac{1}{b - 1}}}{\sum\limits_{k = 1}^c\left( \dfrac{1}{\parallel \bm{x}_i - \bm{m}_k \parallel^2} \right)^{\frac{1}{b - 1}}},\quad i = 1, 2, ..., n,\quad j = 1, 2, ..., c
\end{gathered}$$

**算法步骤：**
1. 设定聚类数目$c$和参数$b$。
2. 初始化各个聚类中心$\bm{m}_j$（参考$C$均值算法部分）
3. 重复下面的运算，直到各个样本的隶属度值稳定：
   1. 用当前的聚类中心，根据隶属度函数$\mu_j(\bm{x})$的计算式对其进行更新。
   2. 用当前的隶属度函数，根据聚类中心$\bm{m}_j$的计算式对其进行更新。

算法收敛时，就得到了各类的聚类中心和各个样本对于各类的隶属度值，从而完成了模糊聚类的划分。如果需要，还可以将模糊聚类结果进行去模糊化，即用一定的规则把模糊聚类划分转化为确定性分类。

### 改进的模糊$C$均值算法
为了去除远离各类聚类中心的野值的影响，人们提出了放松的归一化条件，将所有样本对于各类的隶属度总和放宽至$n$：
$$\sum_{j = 1}^c \mu_j(\bm{x}_i) = n,\quad i = 1, 2, ..., n$$

在此情况下，$\bm{m}_j$的表达式计算仍然不变，为$\bm{m}_j = \dfrac{\sum\limits_{i = 1}^n (\mu_j(\bm{x}_i))^b \bm{x}_i}{\sum\limits_{i = 1}^n (\mu_j(\bm{x}_i))^b}$；而$\mu_j(\bm{x})$的表达式则变为

$$\mu_j(\bm{x}_i) = \dfrac{n\left( \dfrac{1}{\parallel \bm{x}_i - \bm{m}_j \parallel^2} \right)^{\frac{1}{b - 1}}}{\sum\limits_{k = 1}^c\sum\limits_{l = 1}^n\left( \dfrac{1}{\parallel \bm{x}_l - \bm{m}_k \parallel^2} \right)^{\frac{1}{b - 1}}},\quad i = 1, 2, ..., n,\quad j = 1, 2, ..., c$$

最后得到的隶属度值可能大于$1$，可以进行归一化；如果要对结果去模糊化，则可以直接进行处理。

优点：
+ 相较于$C$均值算法具有更好的鲁棒性，可在有野值的情况下得到更好的聚类结果。
+ 同时，最终结果对预先确定的聚类数目更不敏感。

缺点：
+ 对聚类中心的初始值较为敏感。（解决：可以先用确定性$C$均值算法或普通模糊$C$均值算法的结果作为初始聚类中心）
+ 如果迭代中出现某个聚类中心距离某个单样本非常近，最后可能会得到只包含这一个样本的聚类。（解决：可以在$\mu_j(\bm{x})$的运算中增加一个非线性处理，例如使之不小于某个值）

## 分级聚类方法
**算法步骤：**
1. 初始化，每个样本形成一个类。
2. 合并：计算任意两个类之间的距离（或相似性），把距离最小（或相似性最大）的两个类合并为一类，记录下这两个类之间的距离（或相似性），其余类不变。
3. 重复步骤2，直到所有样本被合并到两个类中。

**常用度量：**
+ 最近距离：$$\varDelta(\varGamma_i, \varGamma_j) = \min_{\bm{y} \in \varGamma_i \atop \tilde{\bm{y}} \in \varGamma_j} \delta(\bm{y}, \tilde{\bm{y}})$$即两类中相距最近的样本间距离。
+ 最远距离：$$\varDelta(\varGamma_i, \varGamma_j) = \max_{\bm{y} \in \varGamma_i \atop \tilde{\bm{y}} \in \varGamma_j} \delta(\bm{y}, \tilde{\bm{y}})$$即两类中相距最远的样本间距离。
+ 均值距离：$$\varDelta(\varGamma_i, \varGamma_j) = \delta(\bm{m}_i, \bm{m}_j)$$即两类样本间平均距离。

说明：
+ 算法对样本中的噪声较为敏感，但样本数目的增加可以削弱这种影响。
+ 最终得到的聚类树画法是不唯一的；如$((a, b), (c, d))$和$((b, a), (c, d))$本质一样，不代表前者$b, c$之间相较于后者更接近。

## 自组织映射（SOM）神经网络
### SOM网络结构
所有神经元节点**在同一层上**，在**一个平面上呈现规则排列**。
常见的排列形式包括**方形网格排列**和**蜂窝状排列**。
样本特征向量的每一维都通过一定的权值输入到SOM网络的每一个节点上。

神经元计算节点的功能就是对输入的样本给出响应。输入向量连接到某个节点的权值组成的向量称为该节点的**权值向量**。一个节点对输入样本的**响应强度**，就是**该节点的权值向量与输入向量的匹配程度**，可以用欧氏距离或者内积来计算，如果距离小/内积大则响应强度大。对于一个输入样本，在神经元平面上的所有节点中相应最大的节点称为**获胜节点**。

### SOM学习算法和自组织特性
设$X = \{\bm{x} \in R^d\}$为$d$维样本向量集合，记所有神经元集合为$A$，第$i$个神经元的权值为$\bm{m}_i$。
**具体步骤：**
1. **权值初始化：** 用小随机数初始化权值向量。注意各个节点的初始权值不能相等。
2. 在时刻$t$，按照给定的顺序或随机顺序加入一个样本，记为$\bm{x}(t)$。
3. **计算神经元响应，找到当前获胜节点$c$。** 如果用欧氏距离作为匹配准则，则获胜节点为$$c:\quad \parallel \bm{x}(t) - \bm{m}_c(t) \parallel = \min_{i \in A} \{\parallel \bm{x}(t) - \bm{m}_i(t) \parallel\}$$
4. **权值竞争学习：** 对所有神经元节点，用下述准则更新各自的权值$$\bm{m}_{i}(t + 1) = \bm{m}_{i}(t) + \alpha(t) h_{ci}(t) d(\bm{x}(t), \bm{m}_i(t))$$其中，$\alpha(t)$为步长，$d(\cdot, \cdot)$为两个向量间的欧氏距离，$h_{ci}(t)$为节点$i$和$c$间的**近邻函数**值；对于这一近邻函数，如果采用方形网络结构，则相当于在节点$c$的周围定义一个矩形邻域范围$N_c(t)$，在该邻域内$h_{ci}(t)$为$1$，否则为$0$，即此时权值函数的更新为$$\bm{m}_{i}(t + 1) = \begin{cases}
    \bm{m}_{i}(t) + \alpha(t) d(\bm{x}(t), \bm{m}_i(t)) & i \in N_c(t) \\
    \bm{m}_{i}(t) & i \notin N_c(t) \\
\end{cases},\quad \forall i \in A$$
5. 更新步长$\alpha(t)$和邻域$N_c(t)$，如果达到终止条件则算法停止，否则置$t = t + 1$，转步骤2。

在算法中，终止条件一般为事先设定的迭代次数；为了能让网络更有效地达到自组织状态，步长$\alpha(t)$和邻域$N_c(t)$初始值较大，随着时间$t$增加逐渐减小，到终止时$N_c(t)$收敛至只包含最佳节点本身。
除了矩形邻域外，还可以使用其他邻域函数，如高斯函数等。

#### 自组织现象
对于某个输入样本$\bm{x}$，其对应的最佳响应节点即获胜节点$i$会逐渐趋于固定。将固定下来的获胜节点$i$称为$\bm{x}$的**像**，将样本$\bm{x}$称为神经元节点$i$的**原像**。一个样本只能有一个像，但一个神经元可能有多个原像，也可能没有原像。当学习过程中止后，可以统计每个神经元节点上有多少个原像，即有多少个样本映射到该节点，这个量叫做**像密度**。

SOM网络的**自组织现象**：对样本经过适当学习后，每个样本固定映射到一个像节点，在原本样本空间中距离相近的样本趋向于映射到同一个像节点或者在神经元平面上排列相近的像节点，而且节点的像密度与原空间中的样本密度形成近似的单调关系。

### SOM用于模式识别
如果聚类分析只是将样本集分为较少的几个类，自组织映射网络的使用方式并无明显优势。
改进——**自组织映射分析**（简称**自组织分析**或**SOMA**）**：** 通过自组织学习将样本集映射到神经元平面上，得到各个样本的像和各个节点的原像。在节点平面上统计各个节点的原像数目（像密度），得到一个像密度图（如节点按矩形排列时，由$m \times n$个方格组成），每个方格对应一个神经元节点，方格的灰度值代表密度的相对大小。最后，按照密度图对样本集进行分类。
这种方式无须事先确定聚类数目，也可以更好地适应不同的分布情况。

## 一致聚类方法
**基本思路：** 通过不同的数据抽样和不同的方法进行多次聚类，再对结果进行合并，将在大多数聚类结果中一致的结果作为最终的聚类划分依据。

**具体步骤**：
1. 对原始数据集$D = \{x_1, x_2, ..., x_N\}$进行$S$次**无放回的重采样**，每次采样得到的数据子集为$D^{(1)}, ..., D^{(S)}$。
2. 分别在每一个抽样数据集$D^{(s)}$上运行聚类算法，并尝试不同的聚类类别数目$K$。这里定义**连接矩阵**$M_{(s)}^{(K)}$，表示在$D^{(s)}$上将数据聚类为$K$类时，如果样本$i$和样本$j$两个样本处在同一类中，则$M_{(s)}^{(K)}(i, j) = 1$，否则$M_{(s)}^{(K)}(i, j) = 0$。获得所有$S$次聚类的结果后，定义聚类数为$K$时的**整体一致性矩阵**$M^{(K)}$：$$M^{(K)}(i, j) = \dfrac{\sum\limits_s M_{(s)}^{(K)}(i, j)}{\sum\limits_s I_{(s)}(i, j)}$$其中$I_{(s)}$为**指示矩阵**，当样本$i$和$j$都出现在数据集$D^{(S)}$中时，$I_{(s)}(i, j) = 1$，否则$I_{(s)}(i, j) = 0$。
   + 一致性矩阵$M^{(K)}$为对称矩阵，其元素取值范围为$[0, 1]$，取值为$1$表示每次聚类结果两个样本都在同一类，$0$则反之。我们可以对一致性矩阵进行可视化，每次聚类如果完全一致，则会是若干个矩形块排列在对角线上。
3. 定义$\text{Dist}^{(K)} = (1 - M^{(K)})$作为新的距离度量矩阵，根据此距离进行最终的外层聚类，如使用层次聚类方法，以获得做种的聚类结果。

### 类别数$K$的确定
定义CDF函数：
$$\text{CDF}^{(K)}(t) = \dfrac{\sum\limits_{i < j} I(M^{(K)}(i, j) \leq t)}{\dfrac{N(N - 1)}{2}}$$

其中$t \in [0, 1]$，$I(\cdot)$为指示函数，括号内条件满足取$1$，否则取$0$。
CDF函数表示一致性矩阵$M$中，取值小于阈值$t$的样本对占总样本数量的比例。CDF的值域为$[0, 1]$，且$\text{CDF}^{(K)}(1) = 1$。如果聚类一致性高，则CDF取值在$0$和$1$附近时有较大变化，反之则随着$t$增大缓慢上升。
因此，当$K$不同时，我们可以比较函数曲线的线下面积（AUC）来比较CDF函数间的差异。定义$\text{CDF}^{(K)}$函数的曲线下面积$A(K)$为
$$A(K) = \sum_{i = 2}^{\frac{N(N - 1)}{2}} (x_i - x_{i - 1})\text{CDF}^{(K)}(x_i)$$

其中$\{x_1, x_2, ..., x_{\frac{N(N - 1)}{2}}\}$是对矩阵$M^{(K)}$中元素取值从小到大的排序。
通过仿真数据可知，一般随着$K$的增加$A(K)$的取值明显增大，而当$K$的取值小于实际类别数时，$A(K)$变化不大。定义衡量$A(K)$值变化的指标：
$$\Delta(K) = \begin{cases}
    A(K) & K = 2 \\
    \dfrac{A(K + 1) - \hat{A}^{(K)}}{\hat{A}^{(K)}} & K > 2 \\
\end{cases}$$

其中$\hat{A}^{(K)} = \max_{k \in \{2, ..., K\}} A(k)$，因此可以取$\Delta(K)$趋近稳定前一时刻的$K$为可能的聚类数目。

