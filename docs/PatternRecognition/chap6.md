# 典型的非线性分类器

## 分段线性判别函数
### 分段线性距离分类器
把属于$\omega_i,\ i = 1, 2, ..., c$类的样本区域$R_i$划分为$l_i$个子区域$R_i^l,l = 1, 2, ..., l_i$，每个子类的均值是$\bm{m}_i^l$，对样本$\bm{x}, \omega_i$类的判别函数定义为$$g_i(\bm{x}) = \min_{l = 1, 2, ..., l_i} \parallel \bm{x} - \bm{m}_i^l \parallel$$即本类中离样本最近的子类均值到样本的距离。决策规则为：
> 若$g_k(\bm{x}) = \min\limits_{i = 1, 2, ..., c} g_i(\bm{x})$，则决策$\bm{x} \in \omega_k$

### 一般的分段线性判别函数
上一节是分段线性判别函数的特殊情况，适用于各子类在各维分布基本对称的情形。一般情况下，可以对每个子类建立更一般形式的线性判别函数，即把每个类别划分成$l_i$个子类$$\omega_i = \{\omega_i^1, \omega_i^2, ..., \omega_i^{l_i}\},\quad i = 1, 2, ..., c$$

对每个子类，定义一个线性判别函数
$$g_i^l(\bm{x}) = \bm{w}_i^l \cdot \bm{x} + w_{i0}^l,\quad l = 1, 2, ..., l_i,\quad i = 1, 2, ..., c$$

或增广形式
$$g_i^l(\bm{y}) = \bm{\alpha}_i^l \cdot \bm{y},\quad l = 1, 2, ..., l_i,\quad i = 1, 2, ..., c$$

而类$\omega_i$的分段线性判别函数就定义为
$$g_i(\bm{x}) = \max_{l = 1, 2, ..., l_i}g_i^l(\bm{x}),\quad i = 1, 2, ..., c$$

决策规则为：
> 若$g_k(\bm{x}) = \max\limits_{i = 1, 2, ..., c} g_i(\bm{x})$，则决策$\bm{x} \in \omega_k$

两个相邻的类之间的决策面方程就是两个判别函数相等，即$g_i(\bm{x}) = g_j(\bm{x})$。

分类器设计方法：
1. 人工划定子类的划分方案。可以借助聚类等工具。
2. 已知/假定各类的子类数目，采用如下错误修正法设计分类器的同时获得子类划分：
   1. 已知共有$c$个类别$\omega_i,\ i = 1, 2, ..., c$，已知$\omega_i$需被划分为$l_i$个子类。
   2. 初始化：任意给定各类各子类的权值$\bm{\alpha}_i^l(0),\ l = 1, 2, ..., l_i,\ i = 1, 2, ..., c$，通常可以用较小的随机数。
   3. 在时刻$t$，当前权值为$\bm{\alpha}_i^l(t),\ l = 1, 2, ..., l_i,\ i = 1, 2, ..., c$，考虑某个训练样本$\bm{y}_k \in \omega_j$，找出$\omega_j$中各子类中判别函数最大的子类，记为$m$，即$$\bm{\alpha}_j^m(t)^T\bm{y}_k = \max_{l = 1, 2, ..., l_j}\{\bm{\alpha}_j^l(t)^T \bm{y}_k\}$$考察当前样本权值对$\bm{y}_k$的分类情况：
      1. 若$$\bm{\alpha}_j^m(t)^T \bm{y}_k > \bm{\alpha}_i^l(t)^T \bm{y}_k,\quad \forall i = 1, 2, ..., c,\ i \neq j,\quad l = 1, 2, ..., l_i$$即$\bm{y}_k$分类正确，则$\bm{\alpha}_i^l(t)$保持不变，即$\bm{\alpha}_i^l(t + 1) = \bm{\alpha}_i^l(t),\ l = 1, 2, ..., l_i,\ i = 1, 2, ..., c$
      2. 若对某个$i \neq j$，存在子类$l$使得$$\bm{\alpha}_j^m(t)^T \bm{y}_k \leq \bm{\alpha}_i^l(t)^T \bm{y}_k$$则取$\bm{\alpha}_i^l(t)^T \bm{y}_k$中最大的子类（不妨记作$\omega_i$类的第$n$个子类），对权值进行如下修正：$$\begin{aligned}
        \bm{\alpha}_j^m(t + 1) =& \bm{\alpha}_j^m(t) + \rho_t \bm{y}_k \\
        \bm{\alpha}_i^n(t + 1) =& \bm{\alpha}_i^n(t) - \rho_t \bm{y}_k \\
      \end{aligned}$$其余权值不变。
   4. $t = t + 1$，考察下一个样本，回到第3步。迭代直到算法收敛。
3. 子类数目无法事先确定，可以采用分类树的思想来划分子类和设计分段线性判别函数。

## 二次判别函数
一般形式：
$$\begin{aligned}
   g(\bm{x}) =& \bm{x}^T \bm{Wx} + \bm{w}^T \bm{x} + w_0 \\
   =& \sum_{k = 1}^d w_{kk}x_k^2 + 2\sum_{j = 1}^{d - 1}\sum_{k = j + 1}^d w_{jk}x_j x_k + \sum_{j = 1}^d w_j x_j + w_0
\end{aligned}$$其中$\bm{W}$为$d \times d$实对称矩阵，$\bm{w}$为$d$维向量。该判别函数包含$\dfrac{1}{2}d(d + 3) + 1$个参数。

如果直接求解，参数过多，计算复杂，样本量过少时也难以估计。实际中，人们常常采用参数化的方法来估计二次判别函数，如**假定每一类数据都是正态分布**；此时每一类可定义如下二次判别函数
$$g_i(\bm{x}) = K_i^2 - (\bm{x} - \bm{m}_i)^T \bm{\Sigma}_i^{-1}(\bm{x} - \bm{m}_i)$$

其中$\bm{m}_i$为$\omega_i$类样本均值，$\bm{\Sigma}_i$为$\omega_i$类协方差矩阵，$K_i^2$为一个阈值项，它受协方差矩阵和先验概率的影响。该判别函数就是样本到均值的Mahalanobis距离的平方和固定阈值的比较。样本的均值和方差可采用如下估计：
$$\hat{\bm{m}}_i = \dfrac{1}{N_i} \sum_{j = 1}^{N_i} \bm{x}_j \qquad \hat{\bm{\Sigma}}_i = \dfrac{1}{N_i - 1}\sum_{j = 1}^{N_i}(\bm{x}_j - \hat{\bm{m}}_i)(\bm{x}_j - \hat{\bm{m}}_i)^T$$

如果**两类都近似服从正态分布**，则两类间的判别函数为$g_i(\bm{x}) - g_j(\bm{x}) = 0$，决策规则为：
> 若$g_i(\bm{x}) - g_j(\bm{x}) \lessgtr 0$，则决策$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$。

如果**两类中$\omega_1$分布比较成团（近似正态分布），另一类$\omega_2$较均匀地分布在第一类附近**，则只需要对第一类求解其二次判别函数：$g(\bm{x}) = K_1^2 - (\bm{x} - \bm{m}_1)^T \bm{\Sigma}_1^{-1}(\bm{x} - \bm{m}_1)$，决策规则为：
> 若$g(\bm{x}) \lessgtr 0$，则决策$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$。

## 多层感知器神经网络
### 神经元与感知器
设神经元接收到的信号为$\bm{x} = (x_1, x_2, ..., x_n)^T$，输入信号的权值为$\bm{w} = (w_1, w_2, ..., w_n)^T$，则模型可表示为$$y = \theta(\sum_{i = 1}^n w_ix_i + w_0) = \theta(\bm{w}^T \bm{x} + w_0)$$其中$\theta(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$为**阶跃函数**，有时也可用**符号函数**$\text{sgn}(x) = \begin{cases} 1 & x > 0 \\ -1 & x \leq 0 \end{cases}$替代。

从几何上，感知器神经元就是以$$\sum_{i = 1}^n w_ix_i + w_0 = \bm{w}^T \bm{x} + w_0 = 0$$为超平面将特征空间分为两个区域。

### 反向传播算法
1986年，研究者引入**Sigmoid函数**代替了感知器种先前的阶跃函数；对于0-1阶跃函数，使用**Logistic函数**进行替代：$$f(\alpha) = \dfrac{1}{1 + e^{-\alpha}}$$

而对于符号函数，使用 **Tanh函数（双曲正切函数）** 进行替代：$$f(\alpha) = \tanh \alpha = \dfrac{e^{\alpha} - e^{-\alpha}}{e^{\alpha} + e^{-\alpha}} = \dfrac{2}{1 + e^{-2\alpha}} - 1$$

以Logistic函数为例，替代后的感知器模型为$$y = f(\bm{x}) = \dfrac{1}{1 + e^{-\sum\limits_{i = 1}^n w_ix_i - w_0}}$$为方便计算，常常把$w_0$固定为$1$，从而上式简化为$$y = f(\bm{x}) = \dfrac{1}{1 + e^{-\sum\limits_{i = 1}^n w_ix_i}}$$（这里求和应当为$n + 1$项，但为了方便这里简记为$n$项）

多层感知神经网络是一种可普遍适用的非线性学习机器，可以实现任意复杂的函数映射。接下来使用 **反向传播算法（BP算法）** 来求解具体参数。

表示约定：
+ **输入向量表示：** 输入为$n$维列向量$\bm{x} = (x_1, x_2, ..., x_n)^T$。
+ **层表示：** 用上标$l$代表神经元节点所在的层，输入层记$l = 0$，第一个隐藏层记$l = 1$，以此类推；记总层数为$L$，则输出层$l = L - 1$。
+ **每层神经元表示：** 第$l$层第$i$个神经元的输出记作$x_i^l$，对输入层$x_i^0 = x_i,\ i = 1, 2, ..., n$有$n$个神经元。设输出层节点个数为$m$，而第$l$个隐藏层有$n_l$个神经元。
+ **输出向量表示：** 输出为$m$维列向量$\bm{y} = (y_1, y_2, ..., y_m)^T$。
+ **权值表示：** 第$l$层的权值都用$w_{ij}^l$表示，其中上标$l$表示所在层，下标$ij$表示从第$l - 1$层的节点$i$连接到第$l$层的节点$j$。

做法：在训练开始之前，随机地赋予各权值一定的初值。训练过程中，轮流对网络施加各个训练样本。当某个训练样本作用于神经网络输出端后，利用当前权值计算神经网络的输出，这是一个信号从输入到隐藏层再到输出的过程，称作**前向过程**。考查所得到输出和训练样本已知正确输出的误差，根据误差对输出层权值的偏导数来修正输出层的权值；之后再把误差反向传递到倒数第二层的各节点上，根据误差对这些节点权值的导数修正这些权值，以此类推直到把各层的权值都修正一次。

具体步骤：
1. 确定神经网络的结构，用小随机数进行权值初始化，设训练时间$t = 0$。
2. 从训练集中得到一个训练样本$\bm{x} = (x_1, x_2, ..., x_n)^T \in \mathbb{R}^n$，记其期望输出为$\bm{D} = (d_1, d_2, ..., d_m)^T \in \mathbb{R}^m$。
3. 计算$\bm{x}$在当前神经网络下的实际输出$y_r$，这里使用Logistic函数作为Sigmoid函数。
4. 从输出层开始调整权值：对第$l$层，使用下面的公式修正权值$$w_{ij}^l(t + 1) = w_{ij}^l(t) + \Delta w_{ij}^l(t),\quad j = 1, 2, ..., n_l,\ i = 1, 2, ..., n_{l - 1}$$其中，权值修正项$\Delta w_{ij}^l(t)$满足$$\Delta w_{ij}^l(t) = -\eta\delta_j^lx_i^{l - 1}$$其中$\eta$为学习步长，而
   + 对于输出层（$l = L - 1$），$\delta_j^l$为当前输出与期望输出之误差对权值的导数：$$\delta_j^l = -y_j(1 - y_j)(d_j - y_j),\quad j = 1, 2, ..., m$$
   + 对于中间层，$\delta_j^l$为输出误差反向传播到该层的误差对权值的导数：$$\delta_j^l = x_j^l(1 - x_j^l) \sum_{k = 1}^{n_{l + 1}} \delta_k^{l + 1}w_{jk}^{l + 1}(t),\quad j = 1, 2, ..., n_l$$
       + 注释：对于Logistic函数，$f'(\alpha) = f(\alpha)(1 - f(\alpha))$。其它Sigmoid函数需更换为相应形式。
       + 为避免过早收敛到局部极小点，可以在权值更新时将更新项改为“上一次权值修改方向”和“本次负梯度方向”的加权组合，即$$\Delta w_{ij}^l(t) = \alpha \Delta w_{ij}^l(t - 1) + \eta \delta_j^l x_i^{l - 1}$$
   + 简单记忆：$\delta_j^l$为**向后连接的所有误差乘以权值之和，乘以Sigmoid函数导数代入当前输出得到的值**（如果是输出层，向后连接的就是输出结果）
5. 更新完成后重新计算输出，达到停止条件则终止，否则置$t = t + 1$并返回第2步。

### 多层感知器网络应用于模式识别
1. 二分类问题：一类输出为0，另一类为1；或引入阈值划分。
2. 多分类问题：将$c$类中第$k$类变为$c$维向量：$e_k = (0, ..., 0, 1, 0, ..., 0)^T$，其中$1$在第$k$位；或用从$0$到$c$编码，但这种方式可能会导致目标更加复杂。
3. 特征预处理：为了避免特征取值过大导致神经元无法处理，常需要将特征标准化，如固定在区间$[0, 1]$、$[-1, 1]$范围内，或调整至某个均值/方差等。

## 支持向量机
### 广义线性判别函数
假如对于一维的样本特征$x$，要求$x \in \begin{cases}
   \omega_1 & x < b \text{ 或 } x > a \\
   \omega_2 & b < x < a \\
\end{cases}$，则可以引入二次判别函数$$g(x) = (x - a)(x - b)$$并将决策变为：
> 若$g(x) = (x - a)(x - b) \lessgtr 0$，则$x \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$。

一般地，二次判别函数地形式为$$g(x) = c_0 + c_1 x + c_2 x^2$$

适当选择$x \mapsto \bm{y}$的映射，可以将二次判别函数转化为$\bm{y}$的线性函数
$$g(x) = \bm{a}^T \bm{y} = \sum_{i = 1}^s a_iy_i$$式中$\bm{y} = \begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix} = \begin{pmatrix} 1 \\ x \\ x^2 \end{pmatrix}$，$\bm{a} = \begin{pmatrix} a_1 \\ a_2 \\ a_3 \end{pmatrix} = \begin{pmatrix} c_0 \\ c_1 \\ c_2 \end{pmatrix}$

$g(x) = \bm{a}^T \bm{y}$被称为**广义线性判别函数**，$\bm{a}$被称作**广义权向量**。一般地，对于任意高次判别函数$g(x)$（这时$g(x)$可以看做对任意判别函数作级数展开，然后取其截尾后的逼近），都可以通过适当的变换，化为广义线性判别函数来处理。$\bm{a}^T \bm{y}$虽不是$x$的线性判别函数，但却是$\bm{y}$的线性判别函数，广义线性判别函数可以确定一个过原点的超平面。
然而，这种变换会造成维数大大增加，从而造成“维数灾难”：计算量过大，过于稀疏的点也可能造成病态矩阵导致无法计算。如$n$维特征的二次判别函数，新特征空间维数$N = \dfrac{n(n + 3)}{2}$（理解：$\left(\sum\limits_{i = 1}^n x_i \right)^2 + \left(\sum\limits_{i = 1}^n x_i \right)$的项数）

### 核函数变换和支持向量机
回顾线性支持向量机（非完全线性可分情况），其求解的分类器为
$$f(\bm{x}) = \text{sgn}(\bm{w} \cdot \bm{x} + b) = \text{sgn}(\sum_{i = 1}^n \alpha_i y_i (\bm{x}_i \cdot \bm{x}) + b)$$

其中$\alpha_i$为下列二次优化问题的解
$$\begin{aligned}
  \max_{\bm{\alpha}} \quad &Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum_{i, j = 1}^N \alpha_i\alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad 0 \leq \alpha_i \leq C, \quad i = 1, 2, ..., N
\end{aligned}$$

$b$通过使$$y_j \left(\sum_{i = 1}^n \alpha_i y_i (\bm{x}_i \cdot \bm{x}_j) + b \right) - 1= 0$$成立的$\bm{x}_j$（即**支持向量**）求得。
***

对特征$\bm{x}$进行非线性变换$\bm{z} = \varphi(\bm{x})$，得到新空间里的支持向量机决策函数$$f(\bm{x}) = \text{sgn}(\bm{w}^{\varphi} \cdot \bm{z} + b) = \text{sgn}(\sum_{i = 1}^n \alpha_i y_i (\varphi(\bm{x}_i) \cdot \varphi(\bm{x})) + b)$$

最优化问题则变为
$$\begin{aligned}
  \max_{\bm{\alpha}} \quad &Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum_{i, j = 1}^N \alpha_i\alpha_j y_i y_j (\varphi(\bm{x}_i) \cdot \varphi(\bm{x}_j)) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad 0 \leq \alpha_i \leq C, \quad i = 1, 2, ..., N
\end{aligned}$$

定义支持向量的等式变为
$$y_j \left(\sum_{i = 1}^n \alpha_i y_i (\varphi(\bm{x}_i) \cdot \varphi(\bm{x}_j)) + b \right) - 1= 0$$
***
注意到无论变换具体形式如何，变换对支持向量机的影响均是将原样本特征空间中的内积$(\bm{x}_i \cdot \bm{x}_j)$变成新空间中的内积$(\varphi(\bm{x}_i) \cdot \varphi(\bm{x}_j))$。新特征空间中的内积也是原特征的函数，记作$$K(\bm{x}_i, \bm{x}_j) \overset{\text{def}}{=} (\varphi(\bm{x}_i) \cdot \varphi(\bm{x}_j))$$

从而有决策函数
$$f(\bm{x}) = \text{sgn}(\sum_{i = 1}^n \alpha_i y_i K(\bm{x}_i, \bm{x}_j) + b)$$

最优化问题
$$\begin{aligned}
  \max_{\bm{\alpha}} \quad &Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum_{i, j = 1}^N \alpha_i\alpha_j y_i y_j K(\bm{x}_i, \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad 0 \leq \alpha_i \leq C, \quad i = 1, 2, ..., N
\end{aligned}$$

支持向量定义式
$$y_j \left(\sum_{i = 1}^n \alpha_i y_i K(\bm{x}_i, \bm{x}_j) + b \right) - 1= 0$$

此时我们可以跳过$\varphi(\bm{x})$，直接寻找合适的$K(\bm{x}, \bm{x}')$。事实上，**Mercer条件**给出了需要满足的条件：
+ 对于任意的对称函数$K(\bm{x}, \bm{x}')$，它是某个特征空间中的内积运算的充分必要条件：对于任意$\varphi \not\equiv 0$且$\int \varphi^2(\bm{x}) < \infty$，有$$\int\int K(\bm{x}, \bm{x}') \varphi(\bm{x}) \varphi(\bm{x}') > 0$$

事实上，这一条件可放松为满足如下条件的**正定核**：设$K(\bm{x}, \bm{x}')$为定义在空间$X$上的对称函数，且对任意的训练数据$\bm{x}_1, \bm{x}_2, ..., \bm{x}_m \in X$和实系数$a_1, a_2, ..., a_m \in \mathbb{R}$，都有$$\sum_{i, j}a_i a_j K(\bm{x}_i, \bm{x}_j) \geq 0$$对于满足正定条件的核函数，必定存在一个从$X$空间到内积空间$H$的变换$\varphi$使得$$K(\bm{x}, \bm{x}') = \varphi(\bm{x}) \cdot \varphi(\bm{x}')$$

常用核函数形式：
+ **多项式核函数：** $$K(\bm{x}, \bm{x}') = ((\bm{x} \cdot \bm{x}') + 1)^q$$
+ **径向基（RBF）核函数：** $$K(\bm{x}, \bm{x}') = \exp\left( -\dfrac{\parallel \bm{x} - \bm{x}' \parallel^2}{\sigma^2} \right)$$
+ **Sigmoid函数：** $$K(\bm{x}, \bm{x}') = \tanh(v(\bm{x} \cdot \bm{x}') + c)$$这种支持向量机在参数$v$和$c$满足一定取值条件的情况下等价于包含一个多层感知器网络。

### 多类支持向量机
首先在正则化框架下重新表述支持向量机：
设有样本训练集$\{(\bm{x}_i, y_i),\ i = 1, 2, ..., n\},\ \bm{x}_i \in \mathbb{R}^d$为样本特征，$y_i \in \{-1, 1\}$为样本类别标号。待求函数$f(\bm{x}) = h(\bm{x}) + b,\ h \in H_K$，其中$H_K$为由核函数$K$定义的可再生希尔伯特空间（可理解为类似原版的$\sum\limits_{i = 1}^n \alpha_i y_i K(\bm{x}_i, \bm{x}_j)$）。决策规则为$g(\bm{x}) = \text{sgn}(f(\bm{x}))$。支持向量机求解$f$使其能最小化目标函数
$$\dfrac{1}{n}\sum_{i = 1}^n (1 - y_i f(\bm{x}_i))_+ + \lambda \parallel h \parallel^2_{H_K}$$

其中，第一项为支持向量机原来的目标函数式中的松弛因子项，第二项是对函数复杂性的惩罚，$\lambda$负责调节函数复杂性和训练样本上分类精度之间的平衡。
如果样本的类别标号$y$和待求函数$f(\bm{x})$从标量变为向量，则上述表述可以用于多分类问题。

对$k$类问题，$\bm{y}_i$为一个$k$维向量，如果样本$\bm{x}_i$属于第$j$类，则$\bm{y}_i$的第$j$个分量为$1$，其余分量为$-\dfrac{1}{k - 1}$，从而$\bm{y}_i$的各分量总值为$0$。
待求函数$f(\bm{x}) = (f_1(\bm{x}), f_2(\bm{x}), ..., f_k(\bm{x}))$的各分量和也必须为$0$（即$\sum\limits_{i = 1}^k f_i(\bm{x}) = 0$），且每一个分量都定义在核函数可再生希尔伯特空间中（即$f_j(\bm{x}) = h_j(\bm{x}) + b_j,\ h_j \in H_K$）。
将多个类别编码成这样的向量标签后，多类支持向量机就是求$f(\bm{x}) = (f_1(\bm{x}), f_2(\bm{x}), ..., f_k(\bm{x}))$，使得下列目标函数达到最小：
$$\dfrac{1}{n}\sum_{i = 1}^n \bm{L}(\bm{y_i})\cdot (f(x_i) - \bm{y}_i)_+ + \dfrac{\lambda}{2}\parallel h_j \parallel^2_{H_K}$$

其中$\bm{L}(\bm{y}_i)$为损失矩阵$\bm{C}$和样本类别$\bm{y}_i$相对的行向量。损失矩阵$\bm{C}$为一个$k \times k$的矩阵，其对角线为$0$，其余元素均为$1$；假如$\bm{y}_i$的第$j$个分量为$1$，则$\bm{L}(\bm{y}_i)$就取$\bm{C}$的第$j$行。

得到$f(\bm{x})$后，最终的类别决策规则为
$$g(\bm{x}) = \argmax_{j} f_j(\bm{x})$$

### 支持向量回归
考查用线性回归函数$f(\bm{x}) = \bm{w} \cdot \bm{x} + b$来拟合训练数据$\{(\bm{x}_i, y_i),\ i = 1, 2, ..., n\},\ \bm{x}_i \in \mathbb{R}^d,\ y_i \in \mathbb{R}$。
首先考虑所有样本都可以在一定的精度$\varepsilon$范围内用线性函数拟合的情况，即
$$\begin{cases}
   y_i - \bm{w} \cdot \bm{x}_i - b \leq \varepsilon, \\
   \bm{w} \cdot \bm{x}_i + b - y_i \leq \varepsilon, \\
\end{cases} \quad i = 1, 2, ..., n$$

与支持向量分类时控制最大化分类间隔类似，这里也要求最小化$\dfrac{1}{2}\parallel \bm{w} \parallel^2$，其对应的是要求回归函数最平坦。从而，我们有用于回归的支持向量机原问题
$$
\begin{aligned}
   \min_{\bm{w}, b} \quad& \dfrac{1}{2} \parallel \bm{w} \parallel^2 \\
   s.t. \quad & \begin{cases}
      y_i - \bm{w} \cdot \bm{x}_i - b \leq \varepsilon, \\
      \bm{w} \cdot \bm{x}_i + b - y_i \leq \varepsilon, \\
   \end{cases} \quad i = 1, 2, ..., n
\end{aligned}
$$

如果允许拟合误差超过$\varepsilon$，则可以与分类时类似地引入松弛因子，只是此处需要对上下两个方向都引入松弛因子，从而问题变成
$$
\begin{aligned}
   \min_{\bm{w}, b} \quad& \dfrac{1}{2} \parallel \bm{w} \parallel^2 + C\sum_{i = 1}^n (\xi_i + \xi_i^*) \\
   s.t. \quad & \begin{cases}
      y_i - \bm{w} \cdot \bm{x}_i - b \leq \varepsilon + \xi_i^*, \\
      \bm{w} \cdot \bm{x}_i + b - y_i \leq \varepsilon + \xi_i, \\
   \end{cases} \\
   & \xi_i \geq 0,\quad \xi_i^* \geq 0,\quad i = 1, 2, ..., n
\end{aligned}
$$

其中，常数$C$控制着对超出误差限样本的惩罚与函数的平坦性之间的折中。
我们可以推出，原问题的对偶问题为
$$
\begin{gathered}
   \begin{aligned}
       \max_{\bm{\alpha}, \bm{\alpha}^*} \quad W(\bm{\alpha}, \bm{\alpha}^*) =& -\varepsilon \sum_{i = 1}^l (\alpha_i^* + \alpha_i) + \sum_{i = 1}^l (\alpha_i^* - \alpha_i) \\
       &-\dfrac{1}{2}\sum_{i, j = 1}^l (\alpha_i^* - \alpha_i)(\alpha_j^* - \alpha_j)(\bm{x}_i \cdot \bm{x}_j)
   \end{aligned} \\
   \begin{gathered}
      s.t. \quad \sum_{i = 1}^l \alpha_i^* = \sum_{i = 1}^l \alpha_i \\
      0 \leq \alpha_i^* \leq C,\quad i = 1, 2, ..., l \\
      0 \leq \alpha_i \leq C,\quad i = 1, 2, ..., l \\
   \end{gathered}
\end{gathered}
$$

回归函数的权值与对偶问题中的系数的关系是
$$\bm{w} = \sum_{i = 1}^l (\alpha_i^* - \alpha_i) \bm{x}_i$$

得到的回归函数为
$$f(\bm{x}) = \bm{w}\cdot \bm{x} + b = \sum_{i = 1}^l (\alpha_i^* - \alpha_i) (\bm{x}_i \cdot \bm{x}) + b^*$$

与分类情况下类似，这里的多数$\alpha_i$和$\alpha_i^*$都为$0$，非零的$\alpha_i$或$\alpha_i^*$（二者不可能同时非零）对应的样本要么落在距离回归函数距离恰为$\varepsilon$的“$\varepsilon$-管道”上（类似于支持向量，且$\alpha_i$或$\alpha_i^*$不等于$C$），要么落在“$\varepsilon$-管道”外（类似于错分样本，且其对应的$\alpha_i$或$\alpha_i^*$等于$C$）。

类似地，也可以用核函数来实现非线性支持向量机函数拟合，得到的拟合函数如
$$f(\bm{x}) = \sum_{i = 1}^l (\alpha_i^* - \alpha_i) K(\bm{x}, \bm{x}_i) + b^*$$其中$\alpha_i^*, \alpha_i,\ i = 1, 2, ..., l$为以下优化问题的解：
$$
\begin{gathered}
   \begin{aligned}
       \max_{\bm{\alpha}, \bm{\alpha}^*} \quad W(\bm{\alpha}, \bm{\alpha}^*) =& -\varepsilon \sum_{i = 1}^l (\alpha_i^* + \alpha_i) + \sum_{i = 1}^l (\alpha_i^* - \alpha_i) \\
       &-\dfrac{1}{2}\sum_{i, j = 1}^l (\alpha_i^* - \alpha_i)(\alpha_j^* - \alpha_j)K(\bm{x}_i, \bm{x}_j)
   \end{aligned} \\
   \begin{gathered}
      s.t. \quad \sum_{i = 1}^l \alpha_i^* = \sum_{i = 1}^l \alpha_i \\
      0 \leq \alpha_i^* \leq C,\quad i = 1, 2, ..., l \\
      0 \leq \alpha_i \leq C,\quad i = 1, 2, ..., l \\
   \end{gathered}
\end{gathered}
$$

## 核函数机器
### 核Fisher判别
回顾Fisher线性判别的原理：寻找最优的投影方向$\bm{w}$使得以下准则最大化
$$\max J_F(\bm{w}) = \max \dfrac{\bm{w}^T \bm{S}_b \bm{w}}{\bm{w}^T \bm{S}_w \bm{w}}$$

其中，$\bm{S}_b$和$\bm{S}_w$分别为类间离散度矩阵和类内离散度矩阵：
$$\begin{aligned}
   \bm{S}_b =& (\bm{m}_1 - \bm{m}_2)(\bm{m}_1 - \bm{m}_2)^T \\
   \bm{S}_w =& \sum_{q = 1, 2} \sum_{x_i \in \omega_q}(\bm{x}_i - \bm{m}_q)(\bm{x}_i - \bm{m}_q)^T \\
\end{aligned}$$

$\bm{m}_1$和$\bm{m}_2$分别是两类的样本均值向量。如果类内离散度矩阵可逆，则Fisher线性判别的解为$$\bm{w} = \bm{S}_w^{-1}(\bm{m}_1 - \bm{m}_2)$$

***
下面考虑通过非线性变换设计非线性的Fisher判别。
对样本进行非线性变换$\bm{x} \mapsto \varPhi(\bm{x}) \in F$。在变换后的空间$F$中，Fisher线性判别的准则变为
$$\max J_F(\bm{w}) = \max \dfrac{\bm{w}^T \bm{S}_b^{\varPhi} \bm{w}}{\bm{w}^T \bm{S}_w^{\varPhi} \bm{w}}$$

变换后的类间离散度矩阵$\bm{S}_b^{\varPhi}$和类内离散度矩阵$\bm{S}_w^{\varPhi}$则为
$$\begin{aligned}
   \bm{S}_b^{\varPhi} =& (\bm{m}_1^{\varPhi} - \bm{m}_2^{\varPhi})(\bm{m}_1^{\varPhi} - \bm{m}_2^{\varPhi})^T \\
   \bm{S}_w^{\varPhi} =& \sum_{q = 1, 2} \sum_{\bm{x}_i \in \omega_q}(\varPhi(\bm{x}_i) - \bm{m}_q^{\varPhi})(\varPhi(\bm{x}_i) - \bm{m}_q^{\varPhi})^T \\
\end{aligned}$$

其中变换后的样本均值$\bm{m}_i^{\varPhi}$为
$$\bm{m}_i^{\varPhi} = \dfrac{1}{l_i} \sum_{j = 1}^{l_i} \varPhi(\bm{x}_j^i)$$

$i \in \{1, 2\}$为样本类序号，$l_i$为第$i$类样本数，$l$为总样本数。

由于直接在变换空间里求解Fisher线性判别的变换复杂、维数过高，我们需要将该问题转化为核函数求解的形式。

根据可再生核希尔伯特空间的有关理论可以知道，上述问题的任何解$\bm{w} \in F$都处在$F$空间中所有训练样本张成的子空间内，即$$\bm{w} = \sum_{j = 1}^l \alpha_j \varPhi(\bm{x}_j)$$

从而可以推出$$\bm{w}^T \bm{m}_i^{\varPhi} = \dfrac{1}{l_i}\sum_{j = 1}^l \sum_{k = 1}^{l_i} \alpha_j k(\bm{x}_j, \bm{x}_k^i) = \bm{\alpha}^T \bm{M}_i$$

其中，核函数$k(\bm{x}_j, \bm{x}_k^i) = (\varPhi(\bm{x}_j), \varPhi(\bm{x}_k^i))$，$(\bm{M}_i)_j \overset{\text{def}}{=} \dfrac{1}{l_i} \sum\limits_{k = 1}^{l_i}k(\bm{x}_j, \bm{x}_k^i)$，$\bm{M}_i = \sum\limits_{j = 1}^l (\bm{M}_i)_j$。

如果令$\bm{\alpha} = (\alpha_1, \alpha_2, ..., \alpha_l)^T$，记$$\bm{M} = (\bm{M}_1 - \bm{M}_2)(\bm{M}_1 - \bm{M}_2)^T$$

再构建$l \times l_j$矩阵$\bm{K}_j$满足$(\bm{K}_j)_{nm} = k(\bm{x}_n, \bm{x}_m^j)$称为**第$j$类的核函数矩阵**，$\bm{I}$为单位矩阵，并令
$$\bm{N} = \sum_{j = 1, 2}\bm{K}_j(\bm{I} - \dfrac{1}{l_j}\bm{I}) \bm{K}_j^T = \dfrac{l_j - 1}{l_j}\sum_{j = 1, 2}\bm{K}_j \bm{K}_j^T$$

则有
$$\begin{gathered}
   \bm{w}^T \bm{S}_b^{\varPhi} \bm{w} = \bm{\alpha}^T \bm{M} \bm{\alpha} \\
   \bm{w}^T \bm{S}_w^{\varPhi} \bm{w} = \bm{\alpha}^T \bm{N} \bm{\alpha} \\
\end{gathered}$$

从而$$J(\bm{\alpha}) = \dfrac{\bm{\alpha}^T \bm{M} \bm{\alpha}}{\bm{\alpha}^T \bm{N} \bm{\alpha}}$$

可知最优解方向$\bm{\alpha} \propto \bm{N}^{-1}(\bm{M}_1 - \bm{M}_2)$。
如果求出原始的投影方向$\bm{w}$则需要显式计算$\varPhi(\bm{x})$，这样就失去了核函数的优势；然而我们只需要计算原空间的样本到Fisher方向上的投影，因此只需要计算
$$\left\langle \bm{w}, \varPhi(\bm{x}) \right\rangle = \sum_{i = 1}^l \bm{\alpha}_i k(\bm{x}_i, \bm{x})$$

通常上述问题可能是病态的，因为$\bm{N}$非正定；此时可以引入一个新的矩阵$$\bm{N}_{\mu} = \bm{N} + \mu \bm{I}$$使得矩阵正定（$\mu$是一个常数），这样做同时还实现了对$\parallel \bm{\alpha} \parallel^2$的控制。

### 中心支持向量机
我们注意到支持向量机对样本噪声和偏离数据分布的野值非常敏感，因此我们引入**中心支持向量机**（简称CSVM）的概念，通过用中心间隔代替边界间隔，从而在样本极少或存在野值的情况下获得更可靠的分类器。

假定有训练样本集$\{(\bm{x}_1, y_1), (\bm{x}_2, y_2), ..., (\bm{x}_N, y_N)\},\ \bm{x}_i \in \mathbb{R}^d,\ y_i \in\{+1, -1\}$。先考虑线性可分的情况，即
$$y_i(\bm{w} \cdot \bm{x}_i + b) > 0,\quad i = 1, 2, ..., n$$

不失一般性，我们引入一个小的常数$\varepsilon > 0$，要求所有样本满足
$$y_i(\bm{w} \cdot \bm{x}_i + b) \geq \varepsilon > 0,\quad i = 1, 2, ..., n$$

计算两类样本的中心，分别记为$\bm{x}^+$和$\bm{x}^-$，计算其到分类超平面的距离
$$
\begin{gathered}
   d^+ = \dfrac{|\bm{w} \cdot \bm{x}^+ + b|}{\parallel \bm{w} \parallel} = \dfrac{y^+(\bm{w} \cdot \bm{x}^+ + b)}{\parallel \bm{w} \parallel} \\
   d^- = \dfrac{|\bm{w} \cdot \bm{x}^- + b|}{\parallel \bm{w} \parallel} = \dfrac{y^-(\bm{w} \cdot \bm{x}^- + b)}{\parallel \bm{w} \parallel} \\
\end{gathered}
$$其中$y^+ = 1, y^- = -1$。
设训练集中两类样本数分别为$n^+$和$n^-$，$n = n^+ + n^-$，则两个类中心到分类面的距离之和可以写为
$$
d = d^+ + d^- = \dfrac{\sum\limits_{i = 1}^n l_i y_i(\bm{w} \cdot \bm{x}_i + b)}{\parallel \bm{w} \parallel}
$$其中对第一类样本$l_i = \dfrac{1}{n^+}$，对第二类样本$l_i = \dfrac{1}{n^-}$。我们定义该距离为**分类超平面的中心分离间隔**。与支持向量机的情况类似，如果要最大化这个间隔，也存在尺度不确定的问题，为此引入约束条件
$$\sum_{i = 1}^n l_i y_i (\bm{w} \cdot \bm{x}_i) = 1$$

在此约束下，最大化中心分离间隔等价于最小化$\parallel \bm{w} \parallel$，因此得到优化问题：
$$
\begin{aligned}
   \min \quad& \dfrac{1}{2}\parallel \bm{w} \parallel^2 \\
   s.t. \quad& \sum_{i = 1}^n l_i y_i (\bm{w} \cdot \bm{x}_i) = 1 \\
   & y_i(\bm{w} \cdot \bm{x}_i + b) \geq \varepsilon > 0,\quad i = 1, 2, ..., n
\end{aligned}
$$

此处，如果我们取$\varepsilon = 0$（即不作限制），得到的解为两类样本中心的垂直平分线，即**最小距离分类器**；而引入离分类面最近的样本距离分类面距离不能小于$\varepsilon > 0$后，实际上是追求最大化中心分离间隔的同时，保证了离分类面最近的样本也有足够的分类间隔。

当训练样本非线性可分时，引入松弛因子$\xi_i > 0$，问题变为
$$
\begin{aligned}
   \min \quad& \dfrac{1}{2}\parallel \bm{w} \parallel^2 + C(\sum_{i = 1}^n \xi_i)\\
   s.t. \quad& \sum_{i = 1}^n l_i y_i (\bm{w} \cdot \bm{x}_i) = 1 \\
   & y_i(\bm{w} \cdot \bm{x}_i + b) + \xi_i \geq \varepsilon > 0,\quad i = 1, 2, ..., n
\end{aligned}
$$

其中$C$为和支持向量机一样的、控制对错分样本惩罚程度的参数。

使用拉格朗日乘子法，可以得到对偶问题
$$
\begin{aligned}
  \max \quad &Q(\bm{\alpha}, \beta) = \sum_{i = 1}^N \varepsilon\alpha_i + \beta - \dfrac{1}{2} \sum_{i, j = 1}^N (\alpha_i + \beta l_i)(\alpha_j + \beta l_j) y_i y_j (\bm{x}_i \cdot \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad 0 \leq \alpha_i \leq C, \quad i = 1, 2, ..., N,\quad \beta > 0
\end{aligned}
$$

用同样算法优化，得到解为
$$
\bm{w}^* = \sum_{i = 1}^n (\alpha_i^* + \beta^*l_i)y_i\bm{x}_i = \sum_{i = 1}^n \alpha_i^* y_i\bm{x}_i + \beta^*(\bm{x}^+ - \bm{x}^-)
$$

可以看到，这一个解由两部分组成，一部分对应着支持向量机的解，另一部分对应着最小距离分类器，这两部分的折中由受$\varepsilon$影响的$\beta^*$决定。

**事实上，由这一性质，我们可以先求解一般的支持向量机，然后用下面的方法直接显式规定支持向量机权值与最小距离分类器权值之间的折中：**
$$\bm{w}^{\text{CSVM}} = (1 - \lambda)\bm{w}^{\text{SVM}} + \lambda(\bm{x}^+ - \bm{x}^-)$$

中心支持向量机同样可以使用核函数方法，但值得注意的是如果不是使用对偶问题而是上式的等价方法求解，$\bm{x}^+$和$\bm{x}^-$需要在核函数变换后的空间内计算，即新样本需要和两类的每个训练样本计算核函数内积后再求均值。
