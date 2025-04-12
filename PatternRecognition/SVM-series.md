# 支持向量机大合集

## 基础支持向量机
### 线性可分
#### 原问题的推导
训练样本集$$\{(\bm{x}_1, y_1), (\bm{x}_2, y_2), ..., (\bm{x}_N, y_N)\},\ \bm{x}_i \in \mathbb{R}^d,\ y_i \in\{+1, -1\}$$（$\bm{x}_i$为原始样本向量而非增广向量），这些样本是线性可分的；

令$g(\bm{x}) = (\bm{w} \cdot \bm{x}) + b$，则**分类决策函数：** $$f(\bm{x}) = \text{sgn}((\bm{w} \cdot \bm{x}) + b) = \text{sgn}(g(\bm{x}))$$

从而类别$y_i \in \{-1, +1\}$。
***
正确分类：
$$\begin{cases}
  (\bm{w} \cdot \bm{x}_i) + b > 0 & y_i = +1 \\
  (\bm{w} \cdot \bm{x}_i) + b < 0 & y_i = -1 \\
\end{cases}$$

由于 **$\bm{w}$和$b$的尺度作正数倍数调整时不会影响分类决策**，我们可以将条件变成
$$\begin{cases}
  (\bm{w} \cdot \bm{x}_i) + b \geqslant 1 & y_i = +1 \\
  (\bm{w} \cdot \bm{x}_i) + b \leqslant -1 & y_i = -1 \\
\end{cases}$$

进行统一：$$y_i ((\bm{w} \cdot \bm{x}_i) + b) \geqslant 1,\quad i = 1, 2, \cdots, N$$

此时
$$\begin{cases}
    g(\bm{x}) = \bm{w} \cdot \bm{x} + b = 1 & \Longleftrightarrow \bm{w} \cdot \bm{x} + (b - 1) = 0 \\
    g(\bm{x}) = \bm{w} \cdot \bm{x} + b = -1 & \Longleftrightarrow \bm{w} \cdot \bm{x} + (b + 1) = 0\\
\end{cases}$$为**边界超平面**；
由两条平行直线的距离公式，可得分类间隔为
$$M = \dfrac{|(b + 1) - (b - 1)|}{\sqrt{\bm{w} \cdot \bm{w}}} = \dfrac{2}{\parallel \bm{w} \parallel}$$

我们希望最大化分类间隔，从而求解超平面的问题变为
$$\begin{aligned}
  \max_{\bm{w}, b} \quad &\dfrac{2}{\parallel \bm{w} \parallel} \\
  s.t. \quad & y_i((\bm{w} \cdot \bm{x}_i) + b) - 1\geqslant 0, \quad i = 1, 2, ..., N\\
\end{aligned}$$

将原问题中的$\max\limits_{\bm{w}, b} \dfrac{2}{\parallel \bm{w} \parallel}$替换为（对后面计算更有利的）$\min\limits_{\bm{w}, b} \dfrac{1}{2}\parallel \bm{w} \parallel^2$，得到
$$\begin{aligned}
  \min_{\bm{w}, b} \quad &\dfrac{1}{2}\parallel \bm{w} \parallel^2 \\
  s.t. \quad & y_i((\bm{w} \cdot \bm{x}_i) + b) - 1\geqslant 0, \quad i = 1, 2, ..., N\\
\end{aligned}$$

#### 拉格朗日乘子法和对偶问题的推导
我们使用**拉格朗日乘子法**：对每个样本引入一个拉格朗日系数
$$\alpha_i \geqslant 0,\quad i = 1, 2, \cdots, N$$

用$\bm{\alpha}$表示所有拉格朗日系数（并用$\bm{w} \cdot \bm{w}$代替$\parallel \bm{w} \parallel^2$以与之后的核函数相结合），得到拉格朗日函数
$$L(\bm{w}, b, \bm{\alpha}) = \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1)$$

分别对$\bm{w}$和$b$求偏导（这里可以看出为何$\parallel \bm{w} \parallel^2$需要带系数$\dfrac{1}{2}$），得到
$$\begin{cases}
    \dfrac{\partial L}{\partial \bm{w}} = \bm{w} - \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i = 0 \\[1em]
    \dfrac{\partial L}{\partial b} = - \sum\limits_{i = 1}^N \alpha_i y_i = 0
\end{cases} \Longrightarrow \begin{cases}
    \bm{w} = \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i \\[1em]
    \sum\limits_{i = 1}^N \alpha_i y_i = 0
\end{cases}$$
***
接下来证明**我们为什么要考虑取$\max\limits_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha})$：** 
令
$$\theta(\bm{w}) = \max_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha}) = \max_{\bm{\alpha}} \left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1) \right)$$

注意到我们解的可行域为$\bm{C} = y_i((\bm{w} \cdot \bm{x}_i) + b) - 1\geqslant 0$；
当$y_i((\bm{w} \cdot \bm{x}_i) + b) - 1 < 0$（即$\bm{x} \notin \bm{C}$在可行域外）时，我们可以取$\alpha_i = +\infty$，从而
$$\theta(\bm{w}) = \max_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha}) = L(\bm{w}, b, +\infty) = +\infty$$

当$y_i((\bm{w} \cdot \bm{x}_i) + b) - 1 \geqslant 0$（即$\bm{x} \in \bm{C}$在可行域内）时，我们可以取$\alpha_i = 0$，从而
$$\theta(\bm{w}) = \max_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha}) = L(\bm{w}, b, 0) = \dfrac{1}{2}\parallel \bm{w} \parallel^2$$即为原优化目标函数；
从而
$$\theta(\bm{w}) = \begin{cases}
    \dfrac{1}{2}\parallel \bm{w} \parallel^2 & \bm{x} \in \text{可行域 } \bm{C} \\[1em]
    +\infty & \bm{x} \notin \text{可行域 } \bm{C} \\
\end{cases}$$

因此，在可行域$\bm{C} = y_i((\bm{w} \cdot \bm{x}_i) + b) - 1\geqslant 0$内最小化$\dfrac{1}{2}\parallel \bm{w} \parallel^2$，
等价于直接最小化$\theta(\bm{w}) = \max\limits_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha})$，即此时**原问题等价于**

$$\min_{\bm{w}, b}\max_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha})$$
***
将解出来的约束条件代入回上面的问题，得到
$$\begin{aligned}
    &\min_{\bm{w}, b} \max_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha}) \\
    =& \min_{\bm{w}, b} \max_{\bm{\alpha}}\left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1) \right) \\
    =& \max_{\bm{\alpha}} \min_{\bm{w}, b}\left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1) \right) \\
    =& \max_{\bm{\alpha}} \min_{\bm{w}, b}\left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N (\alpha_i y_i (\bm{w} \cdot \bm{x}_i) + \alpha_i y_i b - \alpha_i) \right) \\
    =& \max_{\bm{\alpha}} \min_{\bm{w}, b}\left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N (\alpha_i y_i (\bm{w} \cdot \bm{x}_i)) - b\sum_{i = 1}^N \alpha_i y_i + \sum_{i = 1}^N \alpha_i \right) \\
    =& \max_{\bm{\alpha}} \min_{\bm{w}}\left( \sum_{i = 1}^N \alpha_i + \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N (\alpha_i y_i (\bm{w} \cdot \bm{x}_i)) - b\cdot 0\right) \\
    =& \max_{\bm{\alpha}} \left( \sum_{i = 1}^N \alpha_i + \dfrac{1}{2}\sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i \cdot \sum\limits_{j = 1}^N \alpha_j y_j \bm{x}_j - \sum_{i = 1}^N \left(\alpha_i y_i \left(\sum\limits_{j = 1}^N \alpha_j y_j \bm{x}_j \cdot \bm{x}_i \right) \right) \right) \\
    =& \max_{\bm{\alpha}} \left( \sum_{i = 1}^N \alpha_i + \dfrac{1}{2}\sum\limits_{i = 1}^N \sum\limits_{j = 1}^N \alpha_i \alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j) - \sum\limits_{i = 1}^N \sum\limits_{j = 1}^N \alpha_i \alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j) \right) \\
    =& \max_{\bm{\alpha}} \left( \sum_{i = 1}^N \alpha_i - \dfrac{1}{2}\sum\limits_{i = 1}^N \sum\limits_{j = 1}^N \alpha_i \alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j)  \right)
\end{aligned}$$

从而原问题转化为对偶问题
$$\begin{aligned}
  \max_{\bm{\alpha}} \quad &Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum\limits_{i = 1}^N \sum\limits_{j = 1}^N \alpha_i\alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad \alpha_i \geqslant 0, \quad i = 1, 2, ..., N
\end{aligned}$$

求解出来后，代入$\bm{w}$的表达式$\bm{w} = \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i$得到$\bm{w}$的解；对于$b$，取$\alpha_i > 0$的某一组样本$\bm{x}_k$（这一组样本就是位于边界上的**支持向量**），代入$y_k(\bm{w} \cdot \bm{x}_k + b) - 1 = 0$求解即可。

#### 对偶问题的求解
我们采用**SMO算法**求解：
1. 将所有$\alpha_i$初始化。
2. 选取一对需更新的变量$\alpha_i$和$\alpha_j$。
3. 固定$\alpha_i$和$\alpha_j$以外的参数，求解对偶问题得到更新后的$\alpha_i$和$\alpha_j$。
4. 重复以上步骤，直到$Q(\bm{\alpha})$的值不再增大，或达到迭代次数。

    在只考虑$\alpha_i$和$\alpha_j$时，对偶问题中的约束可以被重写为
    $$\alpha_i y_i + \alpha_j y_j = c$$

    其中$c = -\sum\limits_{k \neq i, j} \alpha_k y_k$为使得原约束条件成立的常数。
    将$\alpha_i y_i + \alpha_j y_j = c$代入对偶问题的优化目标
    $$\max\limits_{\bm{\alpha}} Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum\limits_{i = 1}^N \sum\limits_{j = 1}^N \alpha_i\alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j)$$可以得到一个关于$\alpha_i$的单变量二次规划问题，我们将其简写为以下形式：
    $$\begin{aligned}
    \max\limits_{\alpha_i, \alpha_j}  \quad &Q'(\alpha_i, \alpha_j) = A\alpha_i^2 + B\alpha_j^2 + C\alpha_i \alpha_j + D\alpha_i + E\alpha_j \\
    s.t. \quad &\alpha_i y_i + \alpha_j y_j = c
    \end{aligned}$$

    先将$\alpha_j$写成带$\alpha_i$的形式，即$\alpha_j = \dfrac{c - \alpha_i y_i}{y_j}$，代入$Q'$后求导$\dfrac{\partial Q'}{\partial \alpha_i} \Big|_{\alpha_i} = 0$，得到$\alpha_i$的极值$\alpha_i^*$；

而$\alpha_j \geqslant 0$等价于$\dfrac{c - \alpha_i y_i}{y_j} \geqslant 0$，即$\alpha_i \leqslant \dfrac{c}{y_i}$；观察$\alpha_i^*$是否同时满足$\alpha_i^* \leqslant \dfrac{c}{y_i}$和$\alpha_i^* \geqslant 0$，
如果同时满足则直接取$\alpha_i^*$和$\alpha_j^* = \dfrac{c - \alpha_i^* y_i}{y_j}$作为优化后的解；
如果不同时满足，则剪裁（更新）$\alpha_i^*$：$\alpha_i^* = \begin{cases} \dfrac{c}{y_i} & \alpha_i^* > \dfrac{c}{y_i} \\[1em] 0 & \alpha_i^* < 0 \end{cases}$，再取$\alpha_i^*$和$\alpha_j^* = \dfrac{c - \alpha_i^* y_i}{y_j}$作为优化后的解；

### 线性不可分
#### 原问题的推导
如果样本集并非线性可分，不等式
$$y_i((\bm{w} \cdot \bm{x}_i) + b) - 1\geqslant 0,\quad i = 1, 2, ..., N$$无法被所有样本同时满足；则对于$<0$的样本$\bm{x}_k$，可以在式子左侧添加一个正数$\xi_k$，使得
$$y_k((\bm{w} \cdot \bm{x}_k) + b) - 1 + \xi_k\geqslant 0$$

从这个角度出发，我们给每一个样本都引入一个非负的松弛变量$\xi_i,\ i = 1, 2, ..., N$，从而让不等式约束条件变为$$y_k((\bm{w} \cdot \bm{x}_i) + b) - 1 + \xi_i\geqslant 0, \quad i = 1, 2, ..., N$$对于被正确分类的样本，$\xi_i = 0$；反之，$\xi_i > 0$。

所有样本的松弛因子之和$\sum\limits_{i = 1}^N \xi_i$作为错分程度的反应，数值越大错误程度越大。显然，我们希望$\sum\limits_{i = 1}^N \xi_i$尽量小，因此我们在线性可分的目标函数$\dfrac{1}{2}\parallel \bm{w} \parallel^2$上添加惩罚项，得到广义最优分类面的目标函数
$$\min_{\bm{w}, b} \left( \dfrac{1}{2}\parallel \bm{w} \parallel^2 + C\sum_{i = 1}^N \xi_i \right)$$

在这一情况下，$C$是一个需要人为选择的参数：如果样本线性可分，$C$不会影响结果（松弛因子最后都会变成$0$）；如果样本线性不可分，较小的$C$表示对错误比较容忍而更强调对于正确分类的样本的分类间隔，较大的$C$更强调对分类错误的惩罚。

引入松弛因子后，广义最优分类面的最优化原问题变为：
$$\begin{aligned}
  \min_{\bm{w}, b, \xi_i} \quad &\dfrac{1}{2}\parallel \bm{w} \parallel^2 + C\sum_{i = 1}^N \xi_i\\
  s.t. \quad & y_i((\bm{w} \cdot \bm{x}_i) + b) - 1 + \xi_i\geqslant 0, \quad i = 1, 2, ..., N \\
  &\xi_i \geqslant 0, \quad i = 1, 2, ..., N
\end{aligned}$$

#### 对偶问题的推导
使用拉格朗日乘子法，对每个样本和$\xi_i$分别引入拉格朗日系数
$$\alpha_i \geqslant 0,\quad \gamma_i \geqslant 0, \quad i = 1, 2, \cdots, N$$

用$\bm{\alpha}, \bm{\gamma}$表示所有拉格朗日系数，得到拉格朗日函数

$$\begin{aligned}
    L(\bm{w}, b, \bm{\alpha}, \bm{\gamma}) =& \dfrac{1}{2}(\bm{w} \cdot \bm{w}) + C\sum_{i = 1}^N \xi_i \\
    &- \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1 + \xi_i) - \sum_{i = 1}^N \gamma_i \xi_i
\end{aligned}$$

分别对$\bm{w}, b, \xi_i$求偏导，得到（$\bm{w}, b$的求导结果与最初版本一致的）
$$\begin{cases}
    \dfrac{\partial L}{\partial \bm{w}} = \bm{w} - \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i = 0 \\[1em]
    \dfrac{\partial L}{\partial b} = - \sum\limits_{i = 1}^N \alpha_i y_i = 0 \\[1em]
    \dfrac{\partial L}{\partial \xi_i} = C - \alpha_i - \gamma_i = 0
\end{cases} \Longrightarrow \begin{cases}
    \bm{w} = \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i \\[1em]
    \sum\limits_{i = 1}^N \alpha_i y_i = 0 \\[1em]
    \alpha_i = C - \gamma_i \leqslant C
\end{cases}$$

代入原问题的等价问题$\min\limits_{\bm{w}, b}\max\limits_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha}, \bm{\gamma})$得（与最初版本一致的）
$$\begin{aligned}
    &\min_{\bm{w}, b} \max_{\bm{\alpha}, \bm{\gamma}} L(\bm{w}, b, \bm{\alpha}, \bm{\gamma}) \\
    =& \min_{\bm{w}, b} \max_{\bm{\alpha}, \bm{\gamma}} \left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1) + \sum_{i = 1}^N (C - \alpha_i - \gamma_i)\xi_i\right)\\
    =& \min_{\bm{w}, b} \max_{\bm{\alpha}}\left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - 1) \right) \\
    =& \max_{\bm{\alpha}} \left( \sum_{i = 1}^N \alpha_i - \dfrac{1}{2}\sum\limits_{i = 1}^N \sum\limits_{j = 1}^N \alpha_i \alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j)  \right)
\end{aligned}$$

而$\alpha_i$约束条件包括$\sum\limits_{i = 1}^N y_i\alpha_i = 0$，$\alpha_i \geqslant 0$和$\alpha_i \leqslant C$（原本应当是$\alpha_i = C - \gamma_i \leqslant C$，但$\gamma_i$在优化目标函数中被消掉了，可以任意取值）三个，从而对偶问题为
$$\begin{aligned}
  \max_{\bm{\alpha}} \quad &Q(\bm{\alpha}) = \sum_{i = 1}^N \alpha_i - \dfrac{1}{2} \sum\limits_{i = 1}^N \sum\limits_{j = 1}^N \alpha_i\alpha_j y_i y_j (\bm{x}_i \cdot \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad 0 \leqslant \alpha_i \leqslant C, \quad i = 1, 2, ..., N
\end{aligned}$$

求解仍然使用SMO算法；注意最后“剪裁”时，区间要同时满足$0 \leqslant \alpha_i \leqslant C$和$0 \leqslant \alpha_j \leqslant C$。

求解出来后，代入$\bm{w}$的表达式$\bm{w} = \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i$得到$\bm{w}$的解；对于$b$，取$0 < \alpha_i < C$的某一组样本$\bm{x}_k$（这一组样本就是被正确分类的、位于边界上的**支持向量**），代入$y_k(\bm{w} \cdot \bm{x}_k + b) - 1 = 0$求解即可。

## 中心支持向量机
### 原问题的推导
首先，**用两类样本中心的距离代替原先的边界间隔**；计算两类样本的中心
$$\bm{x}^+ = \dfrac{1}{n^+}\sum_{y_i = 1}\bm{x}_i,\qquad \bm{x}^- = \dfrac{1}{n^-}\sum_{y_i = -1}\bm{x}_i$$

其到分类超平面$\bm{w}\cdot \bm{x} + b = 0$的距离分别为
$$d^+ = \dfrac{|\bm{w}\cdot \bm{x}^+ + b|}{\parallel \bm{w} \parallel} = \dfrac{y^+(\bm{w}\cdot \bm{x}^+ + b)}{\parallel \bm{w} \parallel} \\
d^- = \dfrac{|\bm{w}\cdot \bm{x}^- + b|}{\parallel \bm{w} \parallel} = \dfrac{y^-(\bm{w}\cdot \bm{x}^- + b)}{\parallel \bm{w} \parallel}$$这里$y^+ = 1$，$y^- = -1$；从而，两类中心的距离
$$\begin{aligned}
    d = d^+ + d^- =& \dfrac{\bm{w} \cdot (y^+ \bm{x}^+ + y^- \bm{x}^-) + (y^+ + y^-)b}{\parallel \bm{w} \parallel} \\
    =& \dfrac{\bm{w} \cdot (y^+ \bm{x}^+ + y^- \bm{x}^-)}{\parallel \bm{w} \parallel} \\
    =& \dfrac{\bm{w} \cdot \left(\dfrac{1}{n^+}\sum\limits_{y_i = 1}\bm{x}_i - \dfrac{1}{n^-}\sum\limits_{y_i = -1}\bm{x}_i \right)}{\parallel \bm{w} \parallel} \\
    =& \dfrac{\bm{w} \cdot \left(\dfrac{1}{n^+}\sum\limits_{y_i = 1} y_i\bm{x}_i + \dfrac{1}{n^-}\sum\limits_{y_i = -1}y_i\bm{x}_i \right)}{\parallel \bm{w} \parallel} \\
    =& \dfrac{\sum\limits_{y_i = 1} \dfrac{1}{n^+} y_i (\bm{w} \cdot \bm{x}_i) + \sum\limits_{y_i = -1} \dfrac{1}{n^-} y_i (\bm{w} \cdot \bm{x}_i)}{\parallel \bm{w} \parallel} \\
    =& \dfrac{\sum\limits_{i = 1}^N l_i y_i (\bm{w} \cdot \bm{x}_i)}{\parallel \bm{w} \parallel} = \dfrac{\sum\limits_{i = 1}^N l_i y_i ((\bm{w} \cdot \bm{x}_i) + b)}{\parallel \bm{w} \parallel}\\
\end{aligned}$$

这里$l_i = \begin{cases}
    \dfrac{1}{n^+} & y_i = 1 \\[1em]
    \dfrac{1}{n^-} & y_i = -1 \\
\end{cases}$，而末尾的$b$可加可不加（事实上$b$会在求和的过程中消掉；几何意义上，只要是与分类超平面平行的超平面，其得到的中心距离不变）

同样的，我们需要限制尺度；令$\sum\limits_{i = 1}^N l_i y_i (\bm{w} \cdot \bm{x}_i) = 1$，则最大化$d$等价于最小化$\parallel \bm{w} \parallel$，等价于最小化$\dfrac{1}{2} \parallel \bm{w} \parallel^2$；
再对$y_i(\bm{w} \cdot \bm{x}_i + b) > 0$加上一个小的常数$\varepsilon > 0$，使得$y_i(\bm{w} \cdot \bm{x}_i + b) \geqslant \varepsilon > 0$，这里$\varepsilon$表示**所有样本都至少离分类超平面有一段距离**（代替基本模型中的$1$）；

最终我们得到最优化问题
$$
\begin{aligned}
   \min \quad& \dfrac{1}{2}\parallel \bm{w} \parallel^2 \\
   s.t. \quad& \sum_{i = 1}^n l_i y_i (\bm{w} \cdot \bm{x}_i) = 1 \\
   & y_i(\bm{w} \cdot \bm{x}_i + b) \geqslant \varepsilon > 0,\quad i = 1, 2, ..., N
\end{aligned}
$$

### 对偶问题的推导
使用拉格朗日乘子法，对两个约束条件分别引入拉格朗日系数
$$\alpha_i \geqslant 0, \quad i = 1, 2, \cdots, N,\quad \beta \geqslant 0 $$

用$\bm{\alpha}, \beta$表示所有拉格朗日系数，得到拉格朗日函数

$$\begin{aligned}
    L(\bm{w}, b, \bm{\alpha}, \beta) =& \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N \alpha_i (y_i((\bm{w} \cdot \bm{x}_i) + b) - \varepsilon) \\
    &- \beta \left(\sum_{i = 1}^N l_i y_i (\bm{w} \cdot \bm{x}_i) - 1 \right)
\end{aligned}$$

分别对$\bm{w}, b$求偏导，得到
$$\begin{cases}
    \dfrac{\partial L}{\partial \bm{w}} = \bm{w} - \sum\limits_{i = 1}^N (\alpha_i + \beta l_i)y_i \bm{x}_i = 0 \\[1em]
    \dfrac{\partial L}{\partial b} = - \sum\limits_{i = 1}^N \alpha_i y_i = 0 
\end{cases} \Longrightarrow \begin{cases}
    \bm{w} = \sum\limits_{i = 1}^N (\alpha_i + \beta l_i) y_i \bm{x}_i \\[1em]
    \sum\limits_{i = 1}^N \alpha_i y_i = 0
\end{cases}$$

代入原问题的等价问题$\min\limits_{\bm{w}, b}\max\limits_{\bm{\alpha}} L(\bm{w}, b, \bm{\alpha}, \beta)$得

$$\begin{aligned}
    &\min_{\bm{w}, b} \max_{\bm{\alpha}, \beta} L(\bm{w}, b, \bm{\alpha}, \beta) \\
    =& \min_{\bm{w}, b} \max_{\bm{\alpha}, \beta} \left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N (\alpha_i + \beta l_i) y_i(\bm{w} \cdot \bm{x}_i) - b \sum_{i = 1}^N \alpha_i y_i + \sum_{i = 1}^N \varepsilon \alpha_i + \beta \right) \\
    =& \max_{\bm{\alpha}, \beta} \min_{\bm{w}} \left(\sum_{i = 1}^N \varepsilon \alpha_i + \beta + \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N (\alpha_i + \beta l_i) y_i(\bm{w} \cdot \bm{x}_i) - b \cdot 0 \right) \\
    =& \max_{\bm{\alpha}, \beta} \left(\sum_{i = 1}^N \varepsilon \alpha_i + \beta + \dfrac{1}{2}\sum\limits_{i = 1}^N (\alpha_i + \beta l_i) y_i \bm{x}_i \cdot \sum\limits_{j = 1}^N (\alpha_j + \beta l_j) y_j \bm{x}_j \right. \\
    & \left. - \sum_{i = 1}^N (\alpha_i + \beta l_i) y_i\left(\sum\limits_{j = 1}^N (\alpha_j + \beta l_j) y_j \bm{x}_j \cdot \bm{x}_i\right) \right) \\
    =& \max_{\bm{\alpha}, \beta} \left(\sum_{i = 1}^N \varepsilon \alpha_i + \beta - \dfrac{1}{2} \sum_{i = 1}^N\sum_{j = 1}^N (\alpha_i + \beta l_i)(\alpha_j + \beta l_j) y_i y_j (\bm{x}_i \cdot \bm{x}_j) \right)
\end{aligned}$$

从而对偶问题为
$$\begin{aligned}
  \max_{\bm{\alpha}, \beta} \quad &Q(\bm{\alpha}, \beta) = \sum_{i = 1}^N \varepsilon \alpha_i + \beta - \dfrac{1}{2} \sum_{i = 1}^N\sum_{j = 1}^N (\alpha_i + \beta l_i)(\alpha_j + \beta l_j) y_i y_j (\bm{x}_i \cdot \bm{x}_j) \\
  s.t. \quad &\sum_{i = 1}^N y_i\alpha_i = 0, \quad \alpha_i \geqslant 0, \quad i = 1, 2, ..., N,\quad \beta \geqslant 0
\end{aligned}$$

如果样本**线性不可分**，将约束条件$\alpha_i \geqslant 0$换成$0 \leqslant \alpha_i \leqslant C$即可，$C$的意义和推导与之前一致。

求解出来后，代入$\bm{w}$的表达式$\bm{w} = \sum\limits_{i = 1}^N (\alpha_i + \beta l_i) y_i \bm{x}_i$得到$\bm{w}$的解；对于$b$，取$\alpha_i > 0$的某一组样本$\bm{x}_k$（这一组样本就是位于边界上的**支持向量**），代入$y_k(\bm{w} \cdot \bm{x}_k + b) - \varepsilon = 0$求解即可。

注意到
$$\begin{aligned}
    \bm{w} =& \sum\limits_{i = 1}^N (\alpha_i + \beta l_i) y_i \bm{x}_i = \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i + \beta \sum\limits_{i = 1}^N l_i y_i \bm{x}_i \\
    =& \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i + \beta \left(\sum_{y_i = 1} y_i \cdot \dfrac{1}{n^+} \bm{x}_i + \sum_{y_i = -1} y_i \cdot \dfrac{1}{n^-} \bm{x}_i \right) \\
    =& \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i + \beta \left(\dfrac{1}{n^+} \sum_{y_i = 1} \bm{x}_i - \dfrac{1}{n^-} \sum_{y_i = -1} \bm{x}_i \right) \\
    =& \sum\limits_{i = 1}^N \alpha_i y_i \bm{x}_i + \beta (\bm{x}^+ - \bm{x}^-)
\end{aligned}$$

可见其为支持向量机和最小距离分类器的折中。从而我们可以直接显式规定中心支持向量机的解：

$$\bm{w}^{\text{CSVM}} = (1 - \lambda)\bm{w}^{\text{SVM}} + \lambda(\bm{x}^+ - \bm{x}^-)$$

## 支持向量回归
### 原问题的推导
重新考虑训练样本集$$\{(\bm{x}_1, y_1), (\bm{x}_2, y_2), ..., (\bm{x}_N, y_N)\},\ \bm{x}_i \in \mathbb{R}^d,\ y_i \in \mathbb{R}$$

我们希望用$f(\bm{x}) = \bm{w} \cdot \bm{x} + b$来进行拟合。

首先，假定所有样本都能落在拟合函数两侧，半径为$\varepsilon$的区间内，即
$$\begin{cases}
   y_i - \bm{w} \cdot \bm{x}_i - b \leqslant \varepsilon, \\
   \bm{w} \cdot \bm{x}_i + b - y_i \leqslant \varepsilon, \\
\end{cases} \quad i = 1, 2, ..., n$$

这一区间的总宽度为
$$M = \dfrac{|(b + \varepsilon) - (b - \varepsilon)|}{\sqrt{\bm{w} \cdot \bm{w}}} = \dfrac{2\varepsilon}{\parallel \bm{w} \parallel}$$

我们希望最大化总宽度，实际上等价于最小化$\dfrac{1}{2}\parallel \bm{w} \parallel^2$，则支持向量回归的原问题为

$$
\begin{aligned}
   \min_{\bm{w}, b} \quad& \dfrac{1}{2} \parallel \bm{w} \parallel^2 \\
   s.t. \quad & \begin{cases}
      y_i - \bm{w} \cdot \bm{x}_i - b \leqslant \varepsilon, \\
      \bm{w} \cdot \bm{x}_i + b - y_i \leqslant \varepsilon, \\
   \end{cases} \quad i = 1, 2, ..., n
\end{aligned}
$$

***
如果允许拟合误差超过$\varepsilon$，则可与线性不可分情况时类似地引入松弛因子，使得约束条件变成
$$\begin{aligned}
    &\begin{cases}
        y_i - \bm{w} \cdot \bm{x}_i - b \leqslant \varepsilon + \xi_i^*, \\
        \bm{w} \cdot \bm{x}_i + b - y_i \leqslant \varepsilon + \xi_i, \\
    \end{cases} \quad i = 1, 2, ..., n \\
    &\xi_i \geqslant 0,\quad \xi_i^* \geqslant 0,\quad i = 1, 2, ..., n
\end{aligned}$$

同样，目标函数在我们引入控制常数$C$（这里表示**超出误差限样本的惩罚和函数平坦性之间的折中**）后变为
$$
\dfrac{1}{2} \parallel \bm{w} \parallel^2 + C\sum_{i = 1}^n (\xi_i + \xi_i^*)
$$

从而新的原问题为
$$
\begin{aligned}
   \min_{\bm{w}, b} \quad& \dfrac{1}{2} \parallel \bm{w} \parallel^2 + C\sum_{i = 1}^n (\xi_i + \xi_i^*) \\
   s.t. \quad & \begin{cases}
      y_i - \bm{w} \cdot \bm{x}_i - b \leqslant \varepsilon + \xi_i^*, \\
      \bm{w} \cdot \bm{x}_i + b - y_i \leqslant \varepsilon + \xi_i, \\
   \end{cases} \\
   & \xi_i^* \geqslant 0,\quad \xi_i \geqslant 0,\quad i = 1, 2, ..., n
\end{aligned}
$$

### 对偶问题的推导（允许拟合误差超过$\varepsilon$）
使用拉格朗日乘子法，对从上至下、从左至右四个约束条件分别引入拉格朗日系数
$$\alpha_i^* \geqslant 0,\quad \alpha_i \geqslant 0,\quad \gamma_i^* \geqslant 0,\quad \gamma_i \geqslant 0, \quad i = 1, 2, \cdots, N$$

用$\bm{\alpha}^*, \bm{\alpha}, \bm{\gamma}^*, \bm{\gamma}$表示所有拉格朗日系数，得到拉格朗日函数

$$\begin{aligned}
    L(\bm{w}, b, \bm{\alpha}^*, \bm{\alpha}, \bm{\gamma}^*, \bm{\gamma}) =& \dfrac{1}{2}(\bm{w} \cdot \bm{w}) + C\sum_{i = 1}^n (\xi_i + \xi_i^*) \\
    &+ \sum_{i = 1}^N \alpha_i^* (y_i - \bm{w} \cdot \bm{x}_i - b - (\varepsilon + \xi_i^*)) \\
    &+ \sum_{i = 1}^N \alpha_i (\bm{w} \cdot \bm{x}_i + b - y_i - (\varepsilon + \xi_i)) \\
    &- \sum_{i = 1}^N \gamma_i^* \xi_i^* - \sum_{i = 1}^N \gamma_i \xi_i
\end{aligned}$$

分别对$\bm{w}, b, \xi_i^*, \xi_i$求偏导，得到
$$\begin{cases}
    \dfrac{\partial L}{\partial \bm{w}} = \bm{w} - \sum\limits_{i = 1}^N (\alpha_i^* - \alpha_i) \bm{x}_i = 0 \\[1em]
    \dfrac{\partial L}{\partial b} = \sum\limits_{i = 1}^N (\alpha_i - \alpha_i^*) = 0 \\[1em]
    \dfrac{\partial L}{\partial \xi_i^*} = C - \alpha_i^* - \gamma_i^* = 0 \\[1em]
    \dfrac{\partial L}{\partial \xi_i} = C - \alpha_i - \gamma_i = 0
\end{cases} \Longrightarrow \begin{cases}
    \bm{w} = \sum\limits_{i = 1}^N (\alpha_i^* - \alpha_i) \bm{x}_i \\[1em]
    \sum\limits_{i = 1}^N \alpha_i^* = \sum\limits_{i = 1}^N \alpha_i \\[1em]    
    \alpha_i^* = C - \gamma_i^* \leqslant C \\[1em]
    \alpha_i = C - \gamma_i \leqslant C
\end{cases}$$

代入原问题的等价问题$\min\limits_{\bm{w}, b}\max\limits_{\bm{\alpha}^*, \bm{\alpha}, \bm{\gamma}^*, \bm{\gamma}} L(\bm{w}, b, \bm{\alpha}^*, \bm{\alpha}, \bm{\gamma}^*, \bm{\gamma})$得

$$\begin{aligned}
    &\min\limits_{\bm{w}, b}\max\limits_{\bm{\alpha}^*, \bm{\alpha}, \bm{\gamma}^*, \bm{\gamma}} L(\bm{w}, b, \bm{\alpha}^*, \bm{\alpha}, \bm{\gamma}^*, \bm{\gamma}) \\
    =& \min\limits_{\bm{w}, b}\max\limits_{\bm{\alpha}^*, \bm{\alpha}, \bm{\gamma}^*, \bm{\gamma}} \left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N (\alpha_i^* - \alpha_i)(\bm{w} \cdot \bm{x}_i) - \varepsilon\sum_{i = 1}^N (\alpha_i^* + \alpha_i) + \sum_{i = 1}^N y_i(\alpha_i^* - \alpha_i) \right. \\
    &\left. + \sum_{i = 1}^N (C - \alpha_i^* - \gamma_i^*)\xi_i^* + \sum_{i = 1}^N (C - \alpha_i - \gamma_i)\xi_i + b \sum_{i = 1}^N(\alpha_i - \alpha_i^*) \right) \\
    =& \max\limits_{\bm{\alpha}^*, \bm{\alpha}}\min_{\bm{w}} \left( \dfrac{1}{2}(\bm{w} \cdot \bm{w}) - \sum_{i = 1}^N (\alpha_i^* - \alpha_i)(\bm{w} \cdot \bm{x}_i) - \varepsilon\sum_{i = 1}^N (\alpha_i^* + \alpha_i) + \sum_{i = 1}^N y_i(\alpha_i^* - \alpha_i) \right. \\
    &\left. + 0 + 0 + b \cdot 0 \right) \\
    =& \max\limits_{\bm{\alpha}^*, \bm{\alpha}} \left( \dfrac{1}{2}\left(\sum\limits_{i = 1}^N (\alpha_i^* - \alpha_i) \bm{x}_i \cdot \sum\limits_{j = 1}^N (\alpha_j^* - \alpha_j) \bm{x}_j \right)  \right. \\
    &\left.- \sum_{i = 1}^N (\alpha_i^* - \alpha_i)\left(\sum\limits_{j = 1}^N (\alpha_j^* - \alpha_j) \bm{x}_j \cdot \bm{x}_i\right) - \varepsilon\sum_{i = 1}^N (\alpha_i^* + \alpha_i) + \sum_{i = 1}^N y_i(\alpha_i^* - \alpha_i) \right) \\
    =& \max\limits_{\bm{\alpha}^*, \bm{\alpha}} \left( \dfrac{1}{2}\sum\limits_{i = 1}^N \sum\limits_{j = 1}^N (\alpha_i^* - \alpha_i)(\alpha_j^* - \alpha_j) (\bm{x}_i \cdot \bm{x}_j) \right. \\
    &\left.- \sum\limits_{i = 1}^N \sum\limits_{j = 1}^N (\alpha_i^* - \alpha_i)(\alpha_j^* - \alpha_j) (\bm{x}_i \cdot \bm{x}_j) - \varepsilon\sum_{i = 1}^N (\alpha_i^* + \alpha_i) + \sum_{i = 1}^N y_i(\alpha_i^* - \alpha_i) \right) \\
    =& \max\limits_{\bm{\alpha}^*, \bm{\alpha}} \left(- \varepsilon\sum_{i = 1}^N (\alpha_i^* + \alpha_i) + \sum_{i = 1}^N y_i(\alpha_i^* - \alpha_i) - \dfrac{1}{2}\sum\limits_{i = 1}^N \sum\limits_{j = 1}^N (\alpha_i^* - \alpha_i)(\alpha_j^* - \alpha_j) (\bm{x}_i \cdot \bm{x}_j) \right) \\
\end{aligned}$$

注意到约束条件除了$\sum\limits_{i = 1}^N \alpha_i^* = \sum\limits_{i = 1}^N \alpha_i$之外，$\alpha_i^* = C - \gamma_i^* \leqslant C$和$\alpha_i = C - \gamma_i \leqslant C$中的$\gamma_i$与$\gamma_i^*$已经从优化目标函数中完全消去，从而对偶问题为
$$
\begin{gathered}
   \begin{aligned}
       \max_{\bm{\alpha}, \bm{\alpha}^*} \quad W(\bm{\alpha}, \bm{\alpha}^*) =& -\varepsilon \sum_{i = 1}^l (\alpha_i^* + \alpha_i) + \sum_{i = 1}^l y_i(\alpha_i^* - \alpha_i) \\
       &-\dfrac{1}{2}\sum_{i, j = 1}^l (\alpha_i^* - \alpha_i)(\alpha_j^* - \alpha_j)(\bm{x}_i \cdot \bm{x}_j)
   \end{aligned} \\
   \begin{gathered}
      s.t. \quad \sum_{i = 1}^l \alpha_i^* = \sum_{i = 1}^l \alpha_i \\
      0 \leqslant \alpha_i^* \leqslant C,\quad i = 1, 2, ..., l \\
      0 \leqslant \alpha_i \leqslant C,\quad i = 1, 2, ..., l \\
   \end{gathered}
\end{gathered}
$$

求解出合适的$\bm{\alpha}$和$\bm{\alpha}^*$后，可以代入
$$\bm{w} = \sum_{i = 1}^l (\alpha_i^* - \alpha_i) \bm{x}_i$$

求出$\bm{w}$；

求解$b$的方法：通过选取$0 < \alpha_i < C$或$0 < \alpha_i < C$的一个样本（这样的样本落在“$\varepsilon$-管道”的边界上，而不在边界内（$\alpha_i = 0$或$\alpha_i^* = 0$）或边界外（$\alpha_i = C$或$\alpha_i^* = C$）），代入$y_i - \bm{w} \cdot \bm{x}_i - b = \varepsilon + \xi_i^*$或$\bm{w} \cdot \bm{x}_i + b - y_i = \varepsilon + \xi_i$求出$b$。
