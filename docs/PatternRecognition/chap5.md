# 线性学习机器

## 线性回归
假设有训练样本集$$\{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\},\quad x_j \in \mathbb{R}^{d + 1},\quad y_j \in \mathbb{R}$$

我们设计机器学习模型为$$f(\bm{x}) = w_0 + w_1x_1 + ... + w_dx_d = \sum_{i = 1}^d w_ix_i = \bm{w}^T\bm{x}$$其中$\bm{w} = (w_0, w_1, ..., w_d)^T$为待定参数，则我们希望找到$f(\bm{x})$使得平方误差最小，即$$\min_{\bm{w}} E = \min_{\bm{w}} \dfrac{1}{N} \sum_{j = 1}^N (f(\bm{x}_j) - y_j)^2$$

该目标函数可写为矩阵形式：$$\begin{aligned}
  E(\bm{w}) =& \dfrac{1}{N} \sum_{j = 1}^N (f(\bm{x}_j) - y_j)^2 \\
  =& \dfrac{1}{N} \parallel \bm{X} \bm{w} - \bm{y} \parallel^2 = \dfrac{1}{N} (\bm{X} \bm{w} - \bm{y})^T(\bm{X} \bm{w} - \bm{y})
\end{aligned}$$

其中$\bm{X} = \begin{pmatrix}
  \bm{x}_1^T \\  \vdots \\ \bm{x}_N^T
\end{pmatrix}$，$\bm{y} = \begin{pmatrix}
  y_1 \\ \vdots \\ y_N
\end{pmatrix}$。利用$\dfrac{\partial E(\bm{w})}{\partial \bm{w}} = 0$可以解得
$$\bm{w}^* = (\bm{X}^T \bm{X})^{-1}\bm{X}^T \bm{y}$$其中$\bm{X}$也被称作伪逆矩阵，记作$\bm{X}^+$。

## 线性判别函数的基本概念
线性判别函数：$$g(\bm{x}) = \bm{w}^T \bm{x} + w_0$$其中$\bm{x}$为$d$维特征列向量，又称**样本向量**，$\bm{w}$称为**权向量**。$w_0$为常数，称为**阈值权**。

对于二分类问题的线性分类器，可采取下述决策规则：
令$g(\bm{x}) = g_1(\bm{x}) + g_2(\bm{x})$，则$$\begin{cases}
  \bm{x} \in \omega_1 & g(\bm{x}) > 0 \\
  \bm{x} \in \omega_2 & g(\bm{x}) < 0 \\
  \bm{x}\text{ randomly classified / abandoned}  & g(\bm{x}) = 0 \\
\end{cases}$$

判别函数$g(\bm{x})$可以看成是特征空间中某点$\bm{x}$到超平面的距离的一种代数度量。
如果把$\bm{x}$表示成$$\bm{x} = \bm{x}_p + r \dfrac{\bm{w}}{\parallel \bm{w} \parallel}$$其中$\bm{x}_p$为$\bm{x}$在决策面$H$上的投影，$r$为$\bm{x}$到$H$的垂直距离，$\dfrac{\bm{w}}{\parallel \bm{w} \parallel}$为$\bm{w}$方向上的单位向量。从而，我们有$$g(\bm{x}) = r \parallel \bm{w} \parallel \quad\text{or}\quad r = \dfrac{g(\bm{x})}{\parallel \bm{w} \parallel}$$

如果$\bm{x}$为原点，则$g(\bm{x}) = w_0$，可求得原点到决策面$H$的距离$$r_0 = \dfrac{w_0}{\parallel \bm{w} \parallel}$$

## Fisher线性判别分析
设训练样本集为$\mathscr{X} = \{\bm{x}_1, \bm{x}_2, ..., \bm{x}_N\}$，其中$\omega_1$类的样本为$\mathscr{X}_1 = \{\bm{x}_1^1, \bm{x}_2^1, ..., \bm{x}_{N_1}^1\}$，$\omega_2$类的样本为$\mathscr{X}_2 = \{\bm{x}_1^2, \bm{x}_2^2, ..., \bm{x}_{N_2}^2\}$。我们需要找到一个投影方向$\bm{w}$（也为$d$维列向量），使得投影后样本变成$$y_i = \bm{w}^T \bm{x}_i,\quad i = 1, 2, ..., N$$

+ 在原样本空间中：
  + **类均值向量**（类似均值）为$$\bm{m}_i = \dfrac{1}{N_i} \sum_{\bm{x}_j \in \mathscr{X}_i} \bm{x}_j,\quad i = 1, 2$$
  + 定义各类的**类内离散度矩阵**（类似方差）为$$\bm{S}_i = \sum_{\bm{x}_j \in \mathscr{X}_i} (\bm{x}_j - \bm{m}_i)(\bm{x}_j - \bm{m}_i)^T,\quad i = 1, 2$$
  + **总类内离散度矩阵**为$$\bm{S}_w = \bm{S}_1 + \bm{S}_2$$
  + **类间离散度矩阵**为$$\bm{S}_b = (\bm{m}_1 - \bm{m}_2)(\bm{m}_1 - \bm{m}_2)^T$$
+ 在投影后的一维空间中：
  + 两**类均值**为$$\tilde{m}_i = \dfrac{1}{N_i} \sum_{y_j \in \mathscr{Y}_i} y_j = \dfrac{1}{N_i} \sum_{x_j \in \mathscr{X}_i} \bm{w}^T \bm{x}_j = \bm{w}^T \bm{m}_i,\quad i = 1, 2$$
  + **类内离散度**为$$\tilde{S}_i^2 = \sum_{y_j \in \mathscr{Y}_i} (y_j - \tilde{m}_i)^2,\quad i = 1, 2$$
  + **总类内离散度**为$$\tilde{S}_w = \tilde{S}_1^2 + \tilde{S}_2^2$$
  + **类间离散度**为$$\tilde{S}_b = (\tilde{m}_1 - \tilde{m}_2)^2$$

我们希望找到的投影方向，使得投影以后两类尽可能分开，而各类内部又尽可能聚集，这一目标可以表示成如下准则：$$\max J_F(w) = \max \dfrac{\tilde{S}_b}{\tilde{S}_w} = \max \dfrac{(\tilde{m}_1 - \tilde{m}_2)^2}{\tilde{S}_1^2 + \tilde{S}_2^2}$$该函数称为**Fisher准则函数**。

将Fisher准则函数中的变量代入还原，该准则变为$$\max J_F(\bm{w}) = \max \dfrac{\bm{w}^T \bm{S}_b \bm{w}}{\bm{w}^T \bm{S}_w \bm{w}}$$该函数被称为**广义Rayleigh商**。

由于对$\bm{w}$的幅值调整不影响值的变化，我们可以固定分母为常数$c$，从而最优化问题变为$$\begin{aligned}
  \max \quad & \bm{w}^T \bm{S}_b \bm{w} \\
  s.t. \quad & \bm{w}^T \bm{S}_w \bm{w} = c \neq 0
\end{aligned}$$

用拉格朗日乘子法，由于得到的最优解$\bm{w}^*$长度任意，可以取$$\bm{w}^* = \bm{S}_w^{-1} (\bm{m}_1 - \bm{m}_2)$$作为**Fisher判别准则下最优的投影方向**。（也可直接求$\dfrac{\partial J_F(\bm{w})}{\partial \bm{w}} = \bm{0}$再去掉标量后得到）

如果已知先验概率$P(\omega_1)$、$P(\omega_2)$，则阈值权可按最优贝叶斯决策方向，取$$w_0 = -\dfrac{1}{2}(\bm{m}_1 + \bm{m}_2)^T \bm{S}_w^{-1} (\bm{m}_1 - \bm{m}_2) - \ln \dfrac{P(\omega_2)}{P(\omega_1)}$$

不考虑先验概率的不同，则可取$$w_0 = -\dfrac{1}{2}(\tilde{m}_1 + \tilde{m}_2)$$

在已知先验概率的情况下，最终Fisher线性判别分析的决策规则为

> 如果$g(\bm{x}) = \bm{w}^T\left(\bm{x} - \dfrac{1}{2}(\bm{m}_1 + \bm{m}_2)\right) \lessgtr \log \dfrac{P(\omega_2)}{P(\omega_1)}$，则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$

## 感知器
为讨论方便，将原先的向量$\bm{x}$增加一个取常数的维度得到$\bm{y} = (1, x_1, x_2, ..., x_d)^T$，称为$\bm{x}$**增广的样本向量**。同样，定义**增广的权向量**$\bm{\alpha} = (w_0, w_1, ..., w_d)^T$，线性判别函数变为$$g(\bm{y}) = \bm{\alpha}^T \bm{y}$$决策规则为：如果$g(\bm{x}) \lessgtr 0$，则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$。

**样本可分性：** 对于一组样本，如果存在权向量$\bm{\alpha}$使得对于样本集中的任何一个样本$i = 1, 2..., N$，如果$\bm{y} \in \omega_1$则$\bm{\alpha}^T \bm{y} > 0$，如果$\bm{y} \in \omega_2$则$\bm{\alpha}^T \bm{y} < 0$，则称这组样本或这个样本集是**线性可分的**。

如果定义一个新变量$\bm{y}' = \begin{cases}
  \bm{y} & \bm{y} \in \omega_1 \\
  -\bm{y} & \bm{y} \in \omega_2 \\
\end{cases}$（称为**规范化增广样本向量**），则样本可分性的条件变为$$\bm{\alpha}^T \bm{y}' > 0,\quad i = 1, 2, ..., N$$

对于线性可分的一组样本$\bm{y}_1, \bm{y}_2, ..., \bm{y}_N$，如果一个权向量$\bm{\alpha}^*$满足$\bm{\alpha}^T \bm{y} > 0,\ i = 1, 2, ..., N$，则称$\bm{\alpha}^*$为一个**解向量**，权值空间中所有的解向量组成的区域称为**解区**。

为了找到一个解向量，我们定义所有错分样本的和$$J_P(\bm{\alpha}) = \sum_{\bm{\alpha}^T \bm{y}_k \leq 0}(- \bm{\alpha}^T \bm{y}_k)$$作为对错分样本的惩罚，则当且仅当$J_P(\bm{\alpha}^*) = \min J_P(\bm{\alpha}) = 0$时$\bm{\alpha}^*$为解向量。

我们可以用梯度下降方法迭代求解$$\bm{\alpha}(t + 1) = \bm{\alpha}(t) - \rho_t \nabla J_P(\bm{\alpha})$$其中$$\nabla J_P(\bm{\alpha}) = \dfrac{\partial J_P(\bm{\alpha})}{\partial \bm{\alpha}} = \sum_{\bm{\alpha}^T \bm{y}_k \leq 0}(- \bm{y}_k)$$则迭代修正公式为$$\bm{\alpha}(t + 1) = \bm{\alpha}(t) + \rho_t \sum_{\bm{\alpha}^T \bm{y}_k \leq 0} \bm{y}_k$$

实际应用时，可以每一次选择某个/某一部分样本进行检查，直到所有样本都被正确分类为止；步长$\rho_t$可以固定（如$1$）也可以使用可变步长。

## 最小平方误差判别
如果样本线性不可分，不等式组$\bm{\alpha}^T \bm{y}_i > 0,\ i = 1, 2, ..., N$不可能同时满足。由于直接求解线性不等式组有时并不方便，可以引进一系列待定常数$b_i,\ i = 1, 2, ..., N$使得$$\bm{Y\alpha} = \bm{b}$$其中$$\bm{Y}_{N \times \hat
{d}} = \begin{pmatrix}
  \bm{y}_1^T \\ \vdots \\ \bm{y}_1^T
\end{pmatrix} \qquad \bm{b} = (b_1, b_2, ..., b_N)^T$$其中$\hat{d} = d + 1$为增广样本向量的维数。通常$N > \hat{d}$导致方程组无法求得精确解，误差记为$\bm{e} = \bm{Y \alpha} - \bm{b}$，我们可以求解方程组的最小平方误差解，即$$\begin{aligned}
  \bm{\alpha}^* =& \argmin_{\bm{\alpha}} J_S(\bm{\alpha}) \\
  =& \argmin_{\bm{\alpha}} \parallel \bm{Y\alpha} - \bm{b} \parallel^2 = \argmin_{\bm{\alpha}} \sum_{i = 1}^N(\bm{\alpha}^T \bm{y}_i - b_i)^2
\end{aligned}$$

其求解方法有两种：

+ **伪逆法：** $$\bm{\alpha}^* = (\bm{Y}^T \bm{Y})^{-1} \bm{Y}^T \bm{b} = \bm{Y}^+ \bm{b}$$这之中$\bm{Y}^+ = (\bm{Y}^T \bm{Y})^{-1} \bm{Y}^T$即为$\bm{Y}$的伪逆。
+ **梯度下降法：** 利用$$\bm{\alpha}(t + 1) = \bm{\alpha}(t) - \rho_t \nabla (b_k - \bm{\alpha}(t)^T \bm{y}_k) \bm{y}_k$$其中$\bm{y}_k$为使得$\bm{\alpha}(t)^t \bm{y}_k \neq b_k$的样本。

如果对应同一类样本，$b_i$选择相同的值，则最小平方误差的解等价于Fisher线性判别的解；特别地，如果第一类样本对应的$b_i = \dfrac{N}{N_1}$，第二类样本对应的$b_i = \dfrac{N}{N_2}$，则阈值$w_0 = -\bm{m}^T \bm{w}^*$为样本均值在所得一维判别函数方向上的投影。（$N_1$、$N_2$为第一类和第二类的样本数，$N = N_1 + N_2$，$\bm{m}$为全部样本均值）

如果所有样本$b_i = 1$，则$N \to \infty$时，其解为贝叶斯判别函数$$g_0(\bm{x}) = P(\omega_1 | \bm{x}) - P(\omega_2 | \bm{x})$$的最小平方误差逼近。

## Logistic回归
**Logistic函数**（神经网络中属于一种**Sigmoid函数**）：$$\theta(s) = \dfrac{e^s}{1 + e^s} = \dfrac{1}{1 + e^{-s}}$$

如线性平移后的变体，作为概率变化的一种渐进表示：$$P(y = 1 | x) = \dfrac{e^{w_0 + w_1 x}}{1 + e^{w_0 + w_1 x}}$$其常被简记为$P(y | x)$。

对$P(y | x)$取对数变形，得到$P(y | x)$的**logit函数**：$$\ln\left(\dfrac{P(y | x)}{1 - P(y | x)}\right) = w_0 + w_1x$$

类似可得**多元logit函数**：$$\text{logit}(\bm{x}) = \ln\left(\dfrac{P(Y | \bm{x})}{1 - P(y | \bm{x})}\right) = w_0 + w_1x_1 + ... + w_mx_m$$样本属于$y = 1$类的概率为$$P(y | \bm{x}) = \dfrac{e^{w_0 + w_1x_1 + ... + w_mx_m}}{1 + e^{w_0 + w_1x_1 + ... + w_mx_m}}$$

**Logistic回归**就是用logit函数的对数几率模型来描述样本属于某类的可能性与样本特征之间的关系。得到系数后的决策为：若$\text{logit}(\bm{x}) \lessgtr 0$，则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$。

其学习算法为**最大似然法**：
设共有$N$个独立的训练样本$\{(\bm{x}_1, y_1), (\bm{x}_2, y_2), ..., (\bm{x}_N, y_N)\},\ \bm{x}_i \in \mathbb{R}^d,\ y_i \in\{+1, -1\}$，其中$y_i = +1$表示样本属于所关心类别，$y_i = -1$表示不属于。假设样本的类别从某个位置的概率$f(\bm{x})$中产生出来，即$$P(y | \bm{x}) = \begin{cases}
  f(\bm{x}) & y = +1 \\
  1 - f(\bm{x}) & y = -1 \\
\end{cases}$$

取Logistic函数$h(\bm{x}) = \theta(\bm{w}^T \bm{x})$来估计$f(\bm{x})$，其中$\bm{w}$为Logistic函数中待求参数组成的向量。从而，我们得到模型$h$在每个样本上的似然函数：$$P(y_j | \bm{x}_j) = \begin{cases}
  h(\bm{x}_j) & y_j = +1 \\
  1 - h(\bm{x}_j) & y_j = -1 \\
\end{cases}$$

注意到$\theta(-s) = 1 - \theta(s)$，因此定义函数$$l(h | \bm{x}_i, y_j) = P(y_j | \bm{x}_j, h) = \theta(y_j \bm{w}^T \bm{x}_j)$$将分段函数进行统一。
集中所有样本，得到模型的似然函数
$$L(\bm{w}) = \prod_{j = 1}^N P(y_j | \bm{x}_j) = \prod_{j = 1}^N \theta(y_j \bm{w}^T \bm{x}_j)$$

我们采用梯度下降法，定义目标函数为似然函数的负对数，优化问题为
$$\begin{aligned}
  \min \quad E(\bm{w}) =& -\dfrac{1}{N}\ln(L(\bm{w})) \\
  =& \dfrac{1}{N} \sum_{j = 1}^N \ln \left(\dfrac{1}{\theta(y_j \bm{w}^T \bm{x}_j)}\right) \\
  =& \dfrac{1}{N} \sum_{j = 1}^N \ln(1 + e^{-y_j \bm{w}^T \bm{x}_j})
\end{aligned}$$

得到梯度下降法：

> 1. 记时刻$k = 0$，初始化参数$\bm{w}(0)$；
> 2. 计算目标函数的负梯度方向$$\nabla E = -\dfrac{1}{N} \sum_{j = 1}^N \dfrac{y_j \bm{x}_j}{1 + e^{-y_j \bm{w}^T \bm{x}_j}}$$
> 3. 按步长（更新率）$\eta$更新下一时刻参数$$\bm{w}(k + 1) = \bm{w}(k) - \eta \nabla E$$
> 4. 检查是否达到终止条件，是则输出结果$\bm{w}$，否则回到第2步。

## 最优分类超平面与线性支持向量机
### 最优分类超平面
假定有训练样本集$\{(\bm{x}_1, y_1), (\bm{x}_2, y_2), ..., (\bm{x}_N, y_N)\},\ \bm{x}_i \in \mathbb{R}^d,\ y_i \in\{+1, -1\}$（$\bm{x}_i$为原始样本向量而非增广向量），这些样本是线性可分的；即存在超平面$g(\bm{x}) = \bm{w} \cdot \bm{x} + b = 0$把所有$N$个样本都无错误地分开。

**定义：** 如果一个超平面能个欧将训练样本没有错误地分开，且两类训练样本中离超平面最近的样本与超平面之间的距离是最大的，则把这个超平面称为最优分类超平面。

最优分类超平面定义的分类决策函数为$$f(\bm{x}) = \text{sgn}(g(\bm{x})) = \text{sgn}(\bm{w} \cdot \bm{x} + b)$$

我们已经知道向量$\bm{x}$到分类面$g(\bm{x}) = 0$的距离是$\dfrac{|g(\bm{x})|}{\parallel \bm{w} \parallel}$。
超平面能够无误地分开所有$N$个样本，即要求所有样本满足$$\begin{cases}
  (\bm{w} \cdot \bm{x}_i) + b > 0 & y_i = +1 \\
  (\bm{w} \cdot \bm{x}_i) + b < 0 & y_i = -1 \\
\end{cases}$$

由于$\bm{w}$和$b$的尺度作正数倍数调整时不会影响分类决策，我们可以将条件变成$$\begin{cases}
  (\bm{w} \cdot \bm{x}_i) + b \geq 1 & y_i = +1 \\
  (\bm{w} \cdot \bm{x}_i) + b \leq -1 & y_i = -1 \\
\end{cases}$$

可引入$y_i$简化得$$y_i((\bm{w} \cdot \bm{x}_i) + b)\geq 1,\quad i = 1, 2, ..., N$$

这一条件约束的超平面权值尺度，这种超平面称作**规范化的分类超平面**。$g(\bm{x}) = 1$和$g(\bm{x}) = -1$就是过两类中各自离分类面最近的样本、且与分类面平行的两个边界超平面。
由于限制两类离分类最近的样本$g(\bm{x})$分别为$1$和$-1$，可知分类间隔$M = \dfrac{2}{\parallel \bm{w} \parallel}$。从而，求解超平面的问题变为$$\begin{aligned}
  \min_{\bm{w}, b} \quad &\dfrac{1}{2}\parallel \bm{w} \parallel^2 \\
  s.t. \quad & y_i((\bm{w} \cdot \bm{x}_i) + b) - 1\geq 0, \quad i = 1, 2, ..., N\\
\end{aligned}$$

我们使用拉格朗日乘子法求解，对每个样本引入一个拉格朗日系数$\alpha_i \geq 0,\ i = 1, 2, ..., N$，可以将优化问题转化为$$\min_{\bm{w}, b} \max_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha}) = \dfrac{1}{2} (\bm{w}\cdot\bm{w}) - \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1)$$

由于在最优解处，$L(\bm{w}, b, \bm{\alpha})$对$\bm{w}, b$的偏导均为$0$，从而有在最优解处$$\bm{w}^* = \sum_{i = 1}^N \alpha_i^* y_i \bm{x}_i \qquad \sum_{i = 1}^N \alpha_i^* y_i = 0$$

代入上式，原问题变为
$$\begin{aligned}
  \max_{\bm{\alpha}} \quad &Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum_{i, j = 1}^N \alpha_i\alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad \alpha_i \geq 0, \quad i = 1, 2, ..., N
\end{aligned}$$

这是一个对$\alpha_i \geq 0,\ i = 1, 2, ..., N$的二次优化问题，称为**最优超平面的对偶问题**，而最初的最优化问题为**最优超平面的原问题**。

#### 支持向量机的求解
再观察对偶问题
$$\begin{aligned}
  \max_{\bm{\alpha}} \quad &Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum_{i, j = 1}^N \alpha_i\alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad \alpha_i \geq 0, \quad i = 1, 2, ..., N
\end{aligned}$$

我们采用**SMO算法**求解：
1. 将所有$\alpha_i$初始化。
2. 选取一对需更新的变量$\alpha_i$和$\alpha_j$。
3. 固定$\alpha_i$和$\alpha_j$以外的参数，求解对偶问题得到更新后的$\alpha_i$和$\alpha_j$。
4. 重复以上步骤，直到$Q(\bm{\alpha})$的值不再增大，或达到迭代次数。

在只考虑$\alpha_i$和$\alpha_j$时，对偶问题中的约束可以被重写为
$$\alpha_i y_i + \alpha_j y_j = c$$

其中$c = -\sum\limits_{k \neq i, j} \alpha_k y_k$为使得原约束条件成立的常数。将$\alpha_i y_i + \alpha_j y_j = c$代入对偶问题的优化目标$\max\limits_{\bm{\alpha}} Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum_{i, j = 1}^N \alpha_i\alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j)$中，可以得到一个关于$\alpha_i$的单变量二次规划问题。

***
解出对偶问题$\alpha_i^*,\ i = 1, 2, ..., N$后，可以求出原问题的解$$\bm{w}^* = \sum_{i = 1}^N \alpha_i^* y_i \bm{x}_i \qquad f(\bm{x}) = \text{sgn}((\bm{w}^* \cdot \bm{x}) + b^*)$$

下面求解$b^*$：
根据最优化理论中的库恩塔克条件（Kuhn-Tucker条件/K-T条件），$L(\bm{w}, b, \bm{\alpha})$的极值点应当满足$$\alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1) = 0,\quad i = 1, 2, ..., N$$

回顾$$\begin{aligned}
  \min_{\bm{w}, b} \quad &\dfrac{1}{2}\parallel \bm{w} \parallel^2 \\
  s.t. \quad & y_i((\bm{w} \cdot \bm{x}_i) + b) - 1\geq 0, \quad i = 1, 2, ..., N\\
\end{aligned}$$可见对约束条件中取$>$的样本，$\alpha_i = 0$；而只有约束条件中取$=$的样本，$\alpha_i > 0$，求$\bm{w}^*$的加权求和中也只有这些样本参与求和，这些样本被称作**支持向量**。

对这些支持向量，有$$y_i((\bm{w}^* \cdot \bm{x}_i) + b^*) - 1 = 0$$

代入任何一个支持向量均可求得$b^*$；实际操作时可用所有$\alpha_i \neq 0$的样本求解后取平均。

### 线性不可分状况
如果样本集并非线性可分，不等式$$y_i((\bm{w} \cdot \bm{x}_i) + b) - 1\geq 0,\quad i = 1, 2, ..., N$$无法被所有样本同时满足，则对于$<0$的样本$\bm{x}_k$，可以在式子左侧添加一个正数$\xi_k$，使得$y_k((\bm{w} \cdot \bm{x}_k) + b) - 1 + \xi_k\geq 0$；从这个角度出发，我们给每一个样本都引入一个非负的松弛变量$\xi_i,\ i = 1, 2, ..., N$，从而让不等式约束条件变为$$y_k((\bm{w} \cdot \bm{x}_i) + b) - 1 + \xi_i\geq 0, \quad i = 1, 2, ..., N$$对于被正确分类的样本，$\xi_i = 0$；反之，$\xi_i > 0$。

所有样本的松弛因子之和$\sum\limits_{i = 1}^N \xi_i$作为错分程度的反应，数值越大错误程度越大。显然，我们希望$\sum\limits_{i = 1}^N \xi_i$尽量小，因此我们在线性可分的目标函数$\dfrac{1}{2}\parallel \bm{w} \parallel^2$上添加惩罚项，得到广义最优分类面的目标函数$$\min_{\bm{w}, b} \dfrac{1}{2}\parallel \bm{w} \parallel^2 + C\sum_{i = 1}^N \xi_i$$

在这一情况下，$C$是一个需要人为选择的参数：如果样本线性可分，$C$不会影响结果（松弛因子最后都会变成$0$）；如果样本线性不可分，较小的$C$表示对错误比较容忍而更强调对于正确分类的样本的分类间隔，较大的$C$更强调对分类错误的惩罚。

引入松弛因子后，广义最优分类面的最优化原问题变为：
$$\begin{aligned}
  \min_{\bm{w}, b, \xi_i} \quad &\dfrac{1}{2}\parallel \bm{w} \parallel^2 + C\sum_{i = 1}^N \xi_i\\
  s.t. \quad & y_i((\bm{w} \cdot \bm{x}_i) + b) - 1 + \xi_i\geq 0, \quad i = 1, 2, ..., N \\
  &\xi_i \geq 0, \quad i = 1, 2, ..., N
\end{aligned}$$

其对偶问题为
$$\begin{aligned}
  \max_{\bm{\alpha}} \quad &Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum_{i, j = 1}^N \alpha_i\alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad 0 \leq \alpha_i \leq C, \quad i = 1, 2, ..., N
\end{aligned}$$

解出$\alpha_i^*$后，原问题的解仍然为
$$\bm{w}^* = \sum_{i = 1}^N \alpha_i^* y_i \bm{x}_i \qquad f(\bm{x}) = \text{sgn}((\bm{w}^* \cdot \bm{x}) + b^*)$$

最后，利用$0 < \alpha_i < C$的样本，代入$$y_i((\bm{w}^* \cdot \bm{x}_i) + b^*) - 1 + \xi_i = 0$$解得$b^*$。

## 多类线性分类器
### 多个两类分类器的组合
+ 方法一：**一对多**，英文为one-vs-rest或one-over-all，对$c$个类只需要$c - 1$个二分类器。
+ 方法二：**逐对**，英文为pairwise，对$c$个类需要$\dfrac{(c - 1)c}{2}$个二分类器。
+ 方法三：构建二叉树进行分类。

### 多类线性判别函数
指对$c$个类，设计$c$个判别函数$$g_i(\bm{x}) = \bm{w}_i^T \bm{x} + w_{i0},\quad i = 1, 2, ..., c$$

或者将$\bm{x}$增广为$\bm{y} = (1, \bm{x}^T)^T$，设增广权向量$\bm{\alpha}_i = (w_0, \bm{w}_i^T)^T$，得到决策函数的增广形式$$g_i(\bm{x}) = \bm{\alpha}_i^T \bm{y}$$在决策时，取判别函数最大的作为决策归属类。

对能够线性可分的样本进行求解时，可以用与感知器类似的**逐步修正法**：
1. 任意选择初始向量$\bm{\alpha}_i (0),\ i = 1, 2, ..., c$，置$t = 0$；
2. 考察某个样本$\bm{y}^k \in \omega_i$，如果$\bm{\alpha}_i^T(t)\bm{y}^k > \bm{\alpha}_j(t)^T \bm{y}^k$则所有权向量不变；若存在某个类$j$，使得$\bm{\alpha}_i(t)^T \bm{y}^k \leq \bm{\alpha}_j(t)^T \bm{y}^k$，则选择$\bm{\alpha}_i(t)^T \bm{y}^k$最大的类别$j$，对各类权值进行修正$$\begin{cases}
  \bm{\alpha}_i(t + 1) = \bm{\alpha}_i(t) + \rho_t \bm{y}^k \\
  \bm{\alpha}_j(t + 1) = \bm{\alpha}_j(t) - \rho_t \bm{y}^k \\
  \bm{\alpha}_l(t + 1) = \bm{\alpha}_l(t), & l \neq i, j
\end{cases}$$其中$\rho_t$为步长。
3. 如果所有样本分类正确则停止，否则重复步骤2。

样本线性不可分时，可以通过减小步长强制收敛、引入余量等方法处理。

### 多类Logistic回归与Softmax
Logistic函数$$P(y = 1 | x) = \dfrac{e^{w_0 + w_1 x}}{1 + e^{w_0 + w_1 x}}$$的分子可以看作是对样本属于该类可能性的度量，分母的作用则是将这一可能性归一化为概率。

推广到多类，假设每一类$j$都与一个参数$w_j$的指数判别函数成正比，即$P(y = j|\bm{x}) \propto e^{\bm{w}_j \cdot \bm{x}}$，归一化后得到**软最大（Softmax）函数**：$$P(y = j | \bm{x}) = \dfrac{e^{\bm{w}_j \cdot \bm{x}}}{\sum\limits_{k = 1}^c e^{\bm{w}_k \cdot \bm{x}}}$$
