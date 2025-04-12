# 隐马尔可夫模型和贝叶斯网络

## 贝叶斯网络的基本概念
两个随机变量$X$和$Y$独立 $\Longleftrightarrow$ $p(x, y) = p(x) p(y)$
两个随机变量$X$和$Y$关于$Z$独立 $\Longleftrightarrow$ $p(x, y | z) = p(x | z) p(y | z)$；推论：$p(x | y, z) = p(x | z)$

链式法则：$$\begin{aligned}
    p(x_1, x_2, ..., x_d) =& p(x_1) p(x_2 | x_1) p(x_3 | x_1, x_2) ... p(x_d | x_1, x_2, ..., x_{d - 1}) \\
    =& p(x_1) \prod_{i = 2}^d p(x_i | x_1, x_2, ..., x_{i - 1})
\end{aligned}$$

## 隐马尔可夫模型
变量约定：
| 变量表示 | 说明 |
|-|-|
| $$\bm{Q} = \{q_1, q_2, ..., q_n\}$$ | 模型含有$n$个隐状态 |
| $$\bm{V} = \{v_1, v_2, ..., v_V\}$$ | 观测值的取值范围 |
| $$\bm{A} = [a_{ij}]_{n \times n}$$ | 状态转移概率矩阵，$a_{ij}$表示从状态$i$转到状态$j$的概率，满足对$\forall i$，$\sum\limits_{j = 1}^n a_{ij} = 1$ |
| $$\bm{O} = o_1 o_2 ... o_L$$ | 长度为$L$的观测序列，$o_t$的取值为$\bm{V}$中的某个值 |
| $$\bm{x} = x_1 x_2 ... x_L$$ | 长度为$L$的隐状态序列，$x_t$的取值为$\bm{Q}$中的某个值 |
| $$\bm{E} = [e_{i}(j)]_{n \times V}$$ | 发射概率矩阵，$e_{i}(j) = p(o = v_j \| x = q_i)$表示模型隐状态取值$q_i$时观测到$v_j$的概率，满足对$\forall i$，$\sum\limits_{j = 1}^V e_{i}(j) = 1$ |
| $$\bm{\pi} = [\pi_1, \pi_2, ..., \pi_n]$$ | 初始概率分布，$\pi_i$表示马氏链从该状态起始的概率，满足$\sum\limits_{i = 1}^n \pi_i = 1$ |

给定观测序列$\bm{O}$和隐状态序列$\bm{x}$，联合概率为$$p(\bm{O}, \bm{x}) = \pi_{x_1} e_{x_1}(o_1) \prod_{t = 2}^L e_{x_t}(o_t) a_{(x_{t - 1}, x_t)}$$

### HMM评估问题
**目标：** 给定模型$M$，计算观测序列的似然度$p(\bm{O} | M)$。

如果已知观测序列$\bm{O}$背后的隐状态序列$\bm{x}$，则似然度$$p(\bm{O} | \bm{x}) = \prod_{i = 1}^L p(o_i | x_i)$$

联合概率则满足$$p(\bm{O}, \bm{x}) = p(\bm{O} | \bm{x}) p(\bm{x}) = \prod_{i = 1}^L p(o_i | x_i) (\pi_{x_1} \prod_{i = 2}^L p(x_i | x_{i - 1}))$$

如果不知道模型隐状态序列求解观测序列出现概率$p(\bm{O})$，一种思路把观测序列$\bm{O}$的概率分解为它在各种可能的隐状态序列下的概率之和，即$$p(\bm{O}) = \sum_{\bm{x}} p(\bm{O}, \bm{x}) = \sum_{\bm{x}} p(\bm{O} | \bm{x}) p(\bm{x}) $$然而对于$n$个可能隐状态、长度为$L$的序列，其求和项总数达到$n^L$个，会产生“指数爆炸”。因此，我们希望能重复利用计算中重复出现的一些概率值。

#### 前向算法
我们用$\alpha_t(j)$表示**对于长度为$t$的观测子序列$o_1 o_2 ... o_t$且$t$时刻对应隐变量取值为$q_j$时，所对应该状态的概率**。即：$$\alpha_t(j) = p(o_1 o_2 ... o_t, x_t = q_j)$$

而$\alpha_t(j)$可以通过截止到$t - 1$时刻的概率计算，以此类推有$$\alpha_t(j) = e_j(o_t) \sum_{i = 1}^n \alpha_{t - 1}(i) a_{ij}$$

当$t = L$时，$\alpha_L(j)$即表示观测序列为$\bm{O}$且最终时刻隐变量取值为$q_j$的概率。对$L$时刻所有可能的隐变量取值求和，得：$$p(\bm{O}) = \sum_{i = 1}^n \alpha_L(i)$$

实际求解流程：
1. 定义初值：$\alpha_1 (j) = e_j (o_1) \pi_j,\ j = 1, 2, ..., n$
2. 迭代求解：$\alpha_t(j) = e_j(o_t) \sum\limits_{i = 1}^n \alpha_{t - 1}(i) a_{ij},\ t = 1, 2, ..., L$
3. 终止结果：$p(\bm{O}) = \sum\limits_{i = 1}^n \alpha_L(i)$

#### 后向算法
我们用$\beta_t(j)$表示**给定$t$时刻的隐藏状态取值$x_t = q_j$，观察到后续观测值为$o_{t + 1}o_{t + 2}...o_L$的概率**。即：$$\beta_t(j) = p(o_{t + 1}o_{t + 2}...o_L | x_t = q_j)$$

其状态转移方程为$$\beta_t(j) = \sum_{i = 1}^n a_{ji} e_i (o_{t + 1}) \beta_{t + 1}(i)$$

实际求解流程：
1. 定义初值：$\beta_L(j) = 1,\ j = 1, 2, ..., n$
2. 迭代求解：$\beta_t(j) = \sum\limits_{i = 1}^n a_{ji} e_i (o_{t + 1}) \beta_{t + 1}(i)$
3. 终止结果：$p(\bm{O}) = \sum_{i = 1}^n \pi_i e_i(o_1) \beta_1(i)$

#### 前向后向算法
**两种形式合并：**$$p(\bm{O} | M) = \sum_{i = 1}^n \sum_{j = 1}^n \alpha_t(i) a_{ij} e_j(o_{t + 1}) \beta_{t + 1}(j),\quad t = 1, 2, ..., L - 1$$

以上三种方法时间复杂度均为$O(n^2L)$。

### HMM隐状态推断问题（解码问题）
**目标：** 根据模型参数与观测序列，推断最有可能的隐状态序列。

**方法：维特比（Viterbi）算法**

定义$$v_t(j) = \max_{x_1 x_2 ... x_{t - 1}} p(x_1 x_2 ... x_{t - 1}, o_1 o_2 ... o_t, x_t = q_j | M)$$表示**观测序列为$o_1 o_2 ... o_t$且在$t$时刻隐状态为$q_j$的所有状态路径的最大概率**。
上式的递推形式为$$v_t(j) = \max_{i = 1, 2, ..., n} v_{t - 1}(i) a_{ij} e_j(o_t)$$对应的最大概率隐状态序列为$$pa_t(j) = \argmax_{i = 1, 2, ..., n} v_{t - 1}(i) a_{ij} e_j(o_t)$$

采用迭代算法，每一步记录下当前$t$时刻隐状态$x_t = q_j$有可能的最大概率值$v_t(j)$和取得该最大值对应的隐状态路径$pa_t(j)$，该路径即为依据模型得到的概率最大的隐状态序列。具体流程如下：
1. 定义初值：$v_1(j) = e_j(o_1)\pi_j,\ pa_1(j) = 0,\ j = 1, 2, ..., n$
2. 迭代求解：依次对$j = 1, 2, ..., n$、$t = 1, 2, ..., L$求解：
   + $v_t(j) = \max\limits_{i = 1, 2, ..., n}v_{t - 1}(i) a_{ij} e_j(o_t)$
   + $pa_t(j) = \argmax\limits_{i = 1, 2, ..., n}v_{t - 1}(i) a_{ij} e_j(o_t)$
3. 终止结果：$p^* = \max\limits_{i = 1, 2, ..., n}v_L(i),\ x_L^* = \argmax\limits_{i = 1, 2, ..., n}v_L(i)$
4. 路径回溯：依次对$t = L - 1, L - 2, ..., 2, 1$，求解得 $x_t^* = pa_{t + 1}(x_{t + 1}^*)$

时间复杂度为$O(n^2L)$。
如果考虑终止概率，则需要添加从任意隐状态终止的概率$a_{i0},\ i = 1, 2, ..., n$。此时维特比算法的终止条件变为$p^* = \max\limits_{i = 1, 2, ..., n}v_L(i)a_{i0},\ x_L^* = \argmax\limits_{i = 1, 2, ..., n}v_L(i)a_{i0}$。

### HMM学习问题
**目标：** 已知模型结构、观测取值范围$\bm{V}$、模型隐状态集$\bm{Q}$，根据观测序列$\bm{O}$的样本，学习模型参数$\bm{A}$、$\bm{E}$和$\bm{\pi}$。

如果**已知隐状态序列**，则只需要根据**最大似然估计**来求解：
$$\begin{cases}
    \hat{a}_{ij} = \dfrac{T'_{ij}}{\sum\limits_{j'}^n T'_{ij'}} \\ \\
    \hat{e}_{i}(k) = \dfrac{E'_{i}(k)}{\sum\limits_{k'}^V E'_{i}(k')} \\ \\
    \hat{\pi}_i = \dfrac{S'_i}{\sum\limits_{i'} S'_{i'}} = \dfrac{S'_i}{S} \\
\end{cases}$$其中$T'_{ij}$为隐状态序列中从状态$q_i$迁移进入$q_j$的次数，$E'_{i}(k)$是隐状态为$q_i$时观测值为$v_k$的次数，$S'_i$表示初始状态为$q_i$的次数，$S$为总样本数。

#### EM算法求解HMM学习问题
我们可以先随便猜测一套模型参数（初始值），从而转变成已知模型和观测序列，求解隐状态序列的推断问题（解码问题）；在推断出来隐状态序列后，再把模型参数看做未知，用先前的最大似然估计来求解新的模型参数。重复以上过程，直到参数收敛或达到迭代次数。
这种方法就是**EM算法**（Expectancy Maximization）的一种应用。其中：
+ E步骤：利用模型的现有参数求隐变量取值的期望；
+ M步骤：利用当前对隐变量的估计值，对模型参数进行最大似然估计，更新期望；

## 朴素贝叶斯分类器
假设各个特征的取值只依赖于类别标签，而特征之间是相互独立的，即
$$p(x_l x_k | \omega_i) = p(x_l | \omega_i)p(x_k | \omega_i)$$

从而联合概率可以分级为
$$p(x_1, x_2, ..., x_d, \omega_i) = p(x_1 | \omega_i) p(x_2 | \omega_i) ... p(x_d | \omega_i)$$

各类别的先验概率可以通过统计训练样本中第$i$类样本占总训练样本的比率来进行估计：
$$p(Y = \omega_i) = \dfrac{\sum\limits_{j = 1}^N I(y_i = \omega_i)}{N}$$其中$I(\cdot)$为指示函数，括号中条件满足取$1$，否则取$0$。

对于各个特征的条件概率，可以通过第$i$类样本在该特征上的取值进行估计。对于离散取值的特征，考虑第$k$个特征$x_k$，若其有$S_k$种可能的取值，即$\{v_1, v_2, ..., v_{S_k}\}$，则参数的极大似然估计为：
$$p(x_k = v_l | Y = \omega_k) = \dfrac{\sum\limits_{j = 1}^N I(x_{k}^{(i)} = v_l, y_j = \omega_i)}{\sum\limits_{j = 1}^N I(y_j = \omega_i)} \qquad l = 1, 2, ..., S_k$$其中$x_{k}^{(i)}$表示第$j$个样本的第$k$个特征的取值。

当训练样本过少，或者某些特征取值概率较低时，可能会出现$\sum\limits_{j = 1}^N I(x_{k}^{(i)} = v_l, y_j = \omega_i) = 0$的情况，此时直接将$\hat{p}(x_k = v_l | Y = \omega_k)$设置为$0$可能并不太合理。此处可以加入伪计数（pseudo count）来对概率值进行平滑矫正，这种方法也被称为拉普拉斯平滑（Laplace smoothing）。一种平滑项方法为：
$$\begin{aligned}
    p(Y = \omega_i) =& \dfrac{\sum\limits_{j = 1}^N I(y_i = \omega_i) + 1}{N + C} \\
    p(x_k = v_l | Y = \omega_k) =& \dfrac{\sum\limits_{j = 1}^N I(x_{k}^{(i)} = v_l, y_j = \omega_i) + 1}{\sum\limits_{j = 1}^N I(y_j = \omega_i) + S_k}
\end{aligned}$$其中$C$为类别数，$S_k$为第$k$维特征可能取值数。

对于连续取值的变量特征，可以用正态分布、均匀分布等模型来进行建模和分布参数的估计。

## 在贝叶斯网络上的条件独立性
+ 形式一：**头对头**
    + 结构：$X \rightarrow Z \leftarrow Y$
    + 关系：
        + $p(x, y, z) = p(x)p(y)p(z | x, y)$
        + 当$Z$取值未知的情况下，$X$和$Y$之间的关系被阻断。即$p(x, y) = p(x)p(y)$
        + 已知$Z = z'$，则$p(x, y|z) = \dfrac{p(x)p(y)p(z' | x, y)}{p(z')}$
+ 形式二：**尾对尾**
    + 结构：$X \leftarrow Z \rightarrow Y$
    + 关系：
        + $p(x, y, z) = p(x | z)p(y | z)p(z)$
        + 已知$Z = z'$，则$X$和$Y$之间的关系被阻断。$p(x, y|z') = p(x | z')p(y | z')$
            + 推论：$p(x | y, z) = p(x | z)$
        + 当$Z$取值未知的情况下，$X$和$Y$之间不独立。
+ 形式三：**头对尾**
    + 结构：$X \rightarrow Z \rightarrow Y$
    + 关系：
        + $p(x, y, z) = p(x)p(z | x)p(y | z)$
        + 已知$Z = z'$，则$X$和$Y$之间的关系被阻断。即$p(x, y | z') = p(x | z')p(y | z')$
        
**d-分离：** 对于一条无向路径$P$，其被取值已知的节点集合$E$ d-分离，当且仅当至少满足下面一种情况时成立：
+ $P$包含$X \to Z \to Y$或$Y \to Z \to X$，且$Z \in E$
+ $P$包含$X \leftarrow Z \to Y$，且$Z \in E$
+ $P$包含$X \to Z \leftarrow Y$，且$Z$与$Z$的后继节点$\notin E$

其计算可用深度优先搜索方法，时间复杂度为线性。

**马尔可夫覆盖：** 对于网络中的一个节点$t$，若存在某个集合使得在条件于该节点集合的情况下，$t$节点与网络中其他节点条件独立，则称这些集合中最小的集合为$t$节点的**马尔可夫覆盖**，记作$MB(t)$。
该性质的代数表示：$P(t | MB(t), Y) = P(t | MB(t))$

## 贝叶斯网络模型的学习
### 贝叶斯网络的参数学习
**目标：** 已知网络结构和样本数据集合，估计模型未知参数
变量约定：
| 变量表示 | 说明 |
|-|-|
| $$\bm{D} = \{x_1, x_2, ..., x_n\}$$ | $N$个训练样本的数据集合 |

已知$\bm{D}$的情况下，模型参数$\bm{\theta}$的后验概率表示为：$$p(\bm{\theta} | \bm{D}) = \dfrac{p(\bm{D} | \bm{\theta}) p(\bm{\theta})}{p(\bm{D})}$$

由于$p(\bm{D})$与参数无关，我们只需要求解$p(\bm{D} | \bm{\theta}) p(\bm{\theta})$的最大化问题；同时，对概率密度函数取对数，得到估计$$\hat{\bm{\theta}} = \argmax_{\bm{\theta}}\left( \sum_{i = 1}^N \log p(x_i | \bm{\theta}) + \log p(\bm{\theta}) \right)$$如果模型中不同参数的先验概率相同，则变为参数最大似然估计问题。

**概率密度函数的分解：** 假设贝叶斯网络具有$n$个节点，用$\bm{pa}(t)$表示$t\ (t = 1, 2, ..., n)$节点的父节点集合，$\bm{x}_i$是训练样本中该节点所有取值构成的向量，即$\bm{x}_i = (x_{(i, 1)}, x_{(i, 2)}, ..., x_{(i, n)})^T$

$$\begin{aligned}
    p(\bm{D} | \bm{\theta}) =& \prod_{i = 1}^N p(\bm{x}_i | \bm{\theta}) = \prod_{i = 1}^N \prod_{t = 1}^n p(x_{(i, t)} | x_{(i, \bm{pa}(t))}, \bm{\theta}_t) \\
    =& \prod_{t = 1}^n \left( \prod_{i = 1}^N p(x_{(i, t)} | x_{(i, \bm{pa}(t))}, \bm{\theta}_t) \right) = \prod_{t = 1}^n p(\bm{D}_t | \bm{\theta}_t)
\end{aligned}$$其中$\bm{D}_t$表示$t$与$t$的父节点有关的子数据集，$\bm{\theta}_t$为模型在这一部分中的参数。

我们通常假设参数的先验分布是相互独立的，即$p(\bm{\theta}) = \prod\limits_{t = 1}^n p(\bm{\theta}(t))$
则后验概率整体上可分解为$$p(\bm{\theta} | \bm{D}) \sim \prod_{t = 1}^n p(\bm{D}_t | \bm{\theta}_t) p(\bm{\theta}_t)$$从而我们可以将各个部分的概率密度进行分解后分别进行计算。

#### 离散变量的贝叶斯网络参数学习
假设随机变量为离散数值，且满足多项式分布。对节点$t$，有$K_t$种取值，记其取值为$X_t$，用$c$表示$t$节点的父节点集合$\bm{pa}(t)$的取值状态，$c = 1, 2, ..., q_t$，其中$q_t$表示该父节点所有可能状态取值的总数，即$q_t = \prod\limits_{X_i \in \bm{pa}(t)}K_i$。
节点$t$在父节点处在$c$状态下取值为$k$的条件概率记为$p = (X_t = k | \bm{pa}(t) = c) = \theta_{tck}$。记$X_t$取$K_t$种可能的取值条件概率分布为$\bm{\theta}_{tc} = (\theta_{tc1}, \theta_{tc2}, ..., \theta_{tcK_t})$，其满足$\sum\limits_{k = 1}^{K_t} \theta_{tck} = 1$。
用$N_{tck}$表示$N$个训练样本中$X_t$取值为$k$且其父节点取值为状态$c$的样本个数：
$$N_{tck} = \sum_{i = 1}^N I(X_{(i, t)} = k, X_{(i, \bm{pa}(t))} = c)$$

根据多项式分布的特性，似然函数为$$p(\bm{D}_t | \bm{\theta}_t) = \prod_{c = 1}^{q_t} \prod_{k = 1}^{K_t} \theta_{tck}^{N_{tck}} = \prod_{c = 1}^{q_t} p(D_{tc} | \theta_{tc})$$其中$D_{tc}$表示数据中节点$t$的父节点取值为状态$c$的样本集。

根据上述分析，参数的后验概率可以表示为$$p(\bm{\theta}_t | \bm{D}_t) \propto p(\bm{D}_t | \bm{\theta}_t) p(\bm{\theta}_t) = \prod_{c = 1}^{q_t} p(D_{tc} | \theta_{tc}) p(\theta_{tc})$$

由于狄利克雷分布为多项式分布的共轭分布，假设贝叶斯网络的先验概率服从狄利克雷分布$\theta_{tc} \propto \text{Dir} (\alpha_{tc1}, \alpha_{tc2}, ..., \alpha_{tcK})$，其中$\alpha_{tck},\ k = 1, 2, ..., K$为超参数，即$$p(\theta_{tc}) \propto \prod_{k = 1}^{K_t} \theta_{tck}^{\alpha_{tck} - 1}$$

从而$$p(D_{tc} | \theta_{tc}) p(\theta_{tc}) \propto \prod_{k = 1}^{K_t} \theta_{tck}^{N_{tck} - 1} \prod_{k = 1}^{K_t} \theta_{tck}^{\alpha_{tck} - 1} = \prod_{k = 1}^{K_t} \theta_{tck}^{N_{tck} + \alpha_{tck} - 1}$$

由狄利克雷分布性质，后验概率也应当服从狄利克雷分布，且估计为$$\hat{\theta}_{tck} = \dfrac{N_{tck} + \alpha_{tck}}{\sum\limits_{k'}N_{tck'} + \alpha_{tck'}}$$

可见当训练样本数较少时，模型的参数主要受到先验分布参数的影响；而当训练样本数较多时，模型的参数主要受到训练样本取值的影响。

### 贝叶斯网络的结构学习
用$\bm{G}$表示某个网络结构，则$$p(\bm{G} | \bm{D}) = \dfrac{p(\bm{D} | \bm{G})p(\bm{G})}{p(\bm{D})}$$

同样地，我们希望最大化$p(\bm{D} | \bm{G})p(\bm{G})$；假设对各种可能的网络结构，先验概率$p(\bm{G})$相等，则可简化为最大似然问题$$\bm{G}^* = \argmax_{\bm{G}} p(\bm{D} | \bm{G})$$

在这种情况下，算法倾向于获得复杂的模型，但这会造成过拟合；因此结构学习时可以添加惩罚项，构造如下的打分函数：
$$\text{Score}(\bm{D}, \bm{G}) = -\log(p(\bm{D} | \bm{G})) + \text{Penalty}(\bm{D}, \bm{G})$$从而结构问题变为最小化该目标函数，即
$$\begin{aligned}
    \bm{G}^* =& \argmin_{\bm{G}} \text{Score}(\bm{D}, \bm{G}) \\
    =& \argmin_{\bm{G}} (-\log(p(\bm{D} | \bm{G})) + \text{Penalty}(\bm{D}, \bm{G}))
\end{aligned}$$
