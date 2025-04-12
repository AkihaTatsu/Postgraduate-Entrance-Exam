## 基本处理方法

### 注意事项

+ $()$和$\{\}$的使用区别：
  + 只有在**古典概型**中， **$P$之后**使用**圆括号$()$**表示**某个事件发生的概率**，如$P(A)$为$A$发生的概率
  + 在**其他任何地方**， **$P$之后**都使用**花括号$\{\}$** ，如$P\{X > Y\}$
  + $\max$、$\min$应当使用**花括号$\{\}$**
  + $E$、$D$、$\text{Conv}$等各类数字特征相关均使用**圆括号$()$**

## 随机事件和古典概型

### 基本条件处理

+ 古典概型中**不受任何条件限制的等式 / 不等式**：
  + $A - B = A\bar{B}$
  + $P(AB) \leq P(A)$
  + $P(A \cup B) \geq P(A)$
+ **超几何分布**的一种应用场景：
  + 从包含$M$个白球的$N$个球的口袋里随机抓$n$个球，其中有$k$个白球的概率

### 注意事项

+ 事件是否**独立 / 不相关**的判断不应付诸感性，必须**用计算严格证明**
  + 类似地，诸如**两两独立、相互独立**等也必须计算证明
+ 若$P(AB) = 0$，则$A \in \bar{B}$
  + 注意**无法推出**$P(A\bar{B}) = 0$！
+ **不能推出的错误结论**：
  + 求解出$P(X) = 0$，**不代表**$X = \varnothing$！
  + 求解出$P(X) = P(Y)$，**不代表**$X = Y$！


### 基本概念

+ $A$和$B$**独立** $\Rightarrow$ $P(A)P(B) = P(AB)$
  + 注意**无法推出**$P(AB) = 0$
+ $A$和$B$**互不相容** $\Rightarrow$ $P(AB) = 0$

## 一维随机变量

### 基本条件处理

+ **最大值**、**最小值**相关概率的区分（连续随机变量$X$、$Y$相互独立）：
  + $P\{\min\{X, Y\} \geqslant a\} = P\{X \geqslant a\}P\{Y \geqslant a\} = (1 - F_X(a))(1 - F_Y(a))$
  + $P\{\max\{X, Y\} \leqslant a\} = P\{X \leqslant a\}P\{Y \leqslant a\} = F_X(a)F_Y(a)$
  + $P\{\min\{X, Y\} \leqslant a\} = 1 - P\{X > a\}P\{Y > a\} = 1 - (1 - F_X(a))(1 - F_Y(a))$
  + $P\{\max\{X, Y\} \geqslant a\} = 1 - P\{X < a\}P\{Y < a\} = 1 - F_X(a)F_Y(a)$
+ 对于**独立同分布的$X, Y$** ，$P\{X > Y\}$和$P\{X < Y\}$等的计算方法：
  + 若$X, Y$均为**离散随机变量**，则有$P\{X > Y\} = P\{X < Y\} = \frac{1}{2}(1 - P\{X = Y\}) = \frac{1}{2}\left(1 - \sum\limits_k (P\{X = k\})^2\right)$
  + 若$X, Y$均为**连续随机变量**，则有$P\{X > Y\} = P\{X \geqslant Y\} = P\{X < Y\} = P\{X \leqslant Y\} = \frac{1}{2}$

### 注意事项

+ 书写**分布函数**时，不要忘了**核心区间之外，恒等于$0$或$1$的部分**！
+ 书写**概率密度函数**时，不要忘了**核心区间之外，恒等于$0$的部分**！
+ $B(n, p)$虽为二项分布表示，但 **$B(1, p)$为** （二项分布简化后的）**伯努利分布 / 0-1分布**，即取$1$的概率为$p$，取$0$的概率为$1 - p$！
  + 其**展开形式**为$P\{X = k\} = p^k(1 - p)^{1 - k}, k \in \{0, 1\}, p \in (0, 1)$，效果等价于$P\{X = k\} = \begin{cases} p & k = 1 \\ 1 - p & k = 0 \end{cases}$
+ 必须**严格注意定义**$F(x) = P\{X \leqslant x\}$**而非**$P\{X < x\}$！
  + 这其中还存在一个$P\{X = x\}$的差值
  + **只有在$X$为连续随机变量时**，$F(x) = P\{X \leqslant x\} = P\{X < x\}$
    + 连续随机变量中，$P\{X = x\} \equiv 0$，即**概率密度函数$f(x)$和$P\{X = x\}$不是一回事！**


### 概率函数和分布函数的基本性质

+ 概率函数的性质
  + **分布函数**的三个基本性质：也是成为分布函数的**充要条件**
    + **单调性**：单调不减
    + **右连续性**：$\lim\limits_{\,{\rm d}elta x \to 0}F(x + 0) = F(x)$对定义域上所有$x$成立
    + **有界性**：$F(-\infty) = 0, F(+\infty) = 1$
  + **概率密度函数**的两个基本性质：成为概率密度函数的**充要条件**
    + **非负性**：$f(x) \geqslant 0$
    + **在$\mathbb{R}$上积分为$1$** ：$\int_{-\infty}^{+\infty} f(x) \,{\rm d}x = 1$
  + $\{p_i\}$是**概率分布**等价于$p_i \geqslant 0, \sum\limits_i p_i = 1$
  + **检查是否符合条件时不要投机取巧，一条一条检查**！

### 根据已知分布$X$求解分布$Y$的基本方法

+ 根据**函数$Y = g(X)$**求解$Y$的**概率分布**的**基本方法**：
  + **离散 → 离散**：
    + 对$X$的分布区间$\{x_1, x_2, \cdots, x_n\}$内每一个取值计算$g(x_i)$，并配上相应的概率$p_i$（相同取值要叠加）
  + **连续 → 连续**：
    + 记$X$的**概率密度函数**为$f_X(x)$，有
      + **分布函数定义法**：$F_Y(y) = P\{Y \leqslant y\} = P\{g(X) \leqslant y\} = \int_{g(X) \leqslant y}f_X(x) {\rm d}x$
        + 用此方式**求出分布函数**后，**求导**可得**概率密度函数**
      + **（概率密度函数）公式法**：若$g(x)$在$(a, b)$（区间开闭视题目条件变化；之后的区间开闭和此处相同）上**单调递增 / 递减且可导**，则可求出**同单调性的可导反函数**$x = h(y)$（或**题目给出了这一单调可导反函数**），$f_X(x)$在定义域上对应的值域为$(\alpha, \beta)$（即$y \in (\alpha, \beta)$），则有$$f_Y(y) = \begin{cases}
        f_X[h(y)]\cdot |h'(y)| & \alpha < y < \beta \\
        0 & \text{其他} \\
        \end{cases}$$
        + 用此方式**求出概率密度函数**后，**积分**可得**分布函数**
  + **连续 → 离散**：特别出现在$g(X)$为一个**离散化函数**（如$\lfloor x \rfloor$）时
    + 需要**敲定$g(X)$可能的取值，并对每个取值分别计算出现的概率**

### 根据已知分布$X$求解分布$Y$的特殊情况

+ 已知$F_X(x)$，求解$Y = g(X)$的 **分布函数$F_Y(y)$** ：
  + **求解$P\{g(X) \leqslant y\}$** （不要投机取巧！）
+ 若$F_X(x)$为$X$的分布函数，且$F_X(x)$连续，则 **$Y = F_X(X)$服从均匀分布$U(0, 1)$** 
  + 证明：
    1. 对$P\{F_X(X) \leqslant y\}$，由于$F_X(x) \in [0, 1]$，故$y > 1$时$P\{F_X(X) \leqslant y\} \equiv 1$，$y < 0$时$P\{F_X(X) \leqslant y\} \equiv 0$
    2. 当$0 \leqslant y \leqslant 1$时，令$F_X(x)$的反函数为$F_X^{-1}(y)$，则$F_X^{-1}(y)$和$F_X(x)$同样单调不减且连续，从而$f_Y(y) = f_X[F_X^{-1}(y)]\cdot {F_X^{-1}}'(y)$，$F_Y(y) = \int_{-\infty}^y f_X[F_X^{-1}(y)]\cdot {F_X^{-1}}'(y) \,{\rm d}y = \int_{-\infty}^y f_X[F_X^{-1}(y)] \,{\rm d} F_X^{-1}(y) = F_X[F_X^{-1}(y)] = y$

### 正态分布相关

+ **正态分布的加减法**：
  + 若$X \sim \mathcal{N}(\mu, \sigma_1^2), Y \sim \mathcal{N}(\mu, \sigma_2^2)$，$X, Y$独立，则$X \pm Y = \mathcal{N}(\mu, \sigma_1^2 \pm \sigma_2^2)$
  + 若$X \sim \mathcal{N}(\mu_1, \sigma^2), Y \sim \mathcal{N}(\mu_2, \sigma^2)$，$X, Y$独立，则$X \pm Y = \mathcal{N}(\mu_1 \pm \mu_2, \sigma^2)$
+ 推广：**正态分布的线性组合**
  + $X \sim \mathcal{N}(\mu_1, \sigma_1), Y \sim \mathcal{N}(\mu_2, \sigma_2)$，$X, Y$独立，则
    + $aX + b \sim \mathcal{N}(a\mu_1 + b, a^2\sigma_1)$
    + $aX + bY \sim \mathcal{N}(a\mu_1 + b\mu_2, a^2\sigma_1 + b^2\sigma_2)$
+ 正态分布**概率密度函数**的**最大值**：
  + $x = \mu$时取到，为$\frac{1}{\sqrt{2\pi} \sigma}$
+ 正态分布**参数$\mathcal{N}(\mu, \sigma^2)$**和**分布函数**的关系：
  + **正态分布**为$\mathcal{N}(\mu, \sigma^2)$，则**分布函数$F(x) = \varPhi\left(\frac{x - \mu}{\sigma}\right)$**
  + **分布函数$F(x) = \varPhi(ax + b)$** ，则**正态分布**为$\mathcal{N}\left(-\frac{b}{a}, \frac{1}{a^2}\right)$
+ 正态分布型**积分（如$\int_{-\infty}^x e^{-\frac{t^2}{2}}\,{\rm d}t$）**和**分布函数$\varPhi(x)$** 的关系：$$\int_{-\infty}^x e^{-\frac{t^2}{2}}\,{\rm d}t = \sqrt{2\pi} \varPhi(x), \quad \int_{-\infty}^x e^{-t^2}\,{\rm d}t = \sqrt{\pi} \varPhi(\sqrt{2}x),\quad \int_{-\infty}^x e^{-\frac{t^2}{2\sigma^2}}\,{\rm d}t = \sqrt{2\pi}\sigma\varPhi(\frac{x}{\sigma})$$
+ **正态分布的绝对值**相关：

  |                                                              |             $X \sim \mathcal{N}(\mu, \sigma^2)$              |    $X \sim \mathcal{N}(0, \sigma^2)$     | $X \sim \mathcal{N}(0, 1)$ |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------------------------: | :------------------------: |
  |                           $E(|X|)$                           | $\sigma\sqrt{\frac{2}{\pi}}e^{-\frac{\mu^2}{2\sigma^2}} + \mu\left(2\varPhi\left(\frac{\mu}{\sigma}\right) - 1\right)$ |       $\sigma\sqrt{\frac{2}{\pi}}$       |   $\sqrt{\frac{2}{\pi}}$   |
  | $\begin{aligned} &D(|X|) \\ =& E(X^2) - E(|X|)^2 \\ =& E(X)^2 + D(X) - E(|X|)^2 \end{aligned}$ |                              /                               | $\sigma^2\left(1 - \frac{2}{\pi}\right)$ |    $1 - \frac{2}{\pi}$     |

## 多维随机变量

### 注意事项

+ 对**离散变量的分布律**，建议在**每个取值构成的表格右侧和下侧，写上边缘概率$p_{i.}$（右侧）和$p_{.j}$（下侧）**；最后**右下角要补上$1$**

### 二维随机变量

+ 对**在区域$D$上分布的$(x, y)$** 相关的概率问题，建议先**画出$D$**和**题目所求概率对应$(x, y)$范围**在坐标系上的草图
+ 在**求解边缘概率密度**时，若**联合概率密度$f(x, y)$非零区域较为复杂**，则将其以**将要被积分的变量**为**基准**，**拆分为若干个可以确定上下界的区域**后分别讨论
  + 如：对$|x| < y$，可拆分为$\begin{cases} y > x & x \geqslant 0 \\ y > -x & x < 0\end{cases}$，从而$x$的边缘分布为$\begin{cases} \int_{x}^{+\infty} f(x, y) \,{\rm d}y & x \geqslant 0 \\ \int_{-x}^{+\infty} f(x, y) \,{\rm d}y & x < 0\end{cases}$
+ **二维变量**的**独立**：
  + **分布函数**判别：对任意$x, y$，$F(x, y) = F_X(x) \cdot F_Y(y)$，这里$F_X$和$F_Y$为边缘分布函数
  + **密度函数**判别：
    + 对二维**离散随机变量**，**判别式**为$p_{ij} = p_{i.}p_{.j}$
    + 对二维**连续随机变量**，**判别式**为$f(x, y) = f_X(x)f_Y(y)$，这里$f_X$和$f_Y$为边缘密度函数
+ 如果**二维变量$(X, Y)$中$X$和$Y$相互独立**，**则$f_{X \mid Y}(x \mid y) = f_X(x)$**
  + 证明：$f_{X \mid Y}(x \mid y) = \frac{f(x, y)}{f_Y(y)} = \frac{f_X(x)f_Y(y)}{f_Y(y)} = f_X(x)$
+ 二维随机变量**分布函数**和**概率密度函数**的**线性变化关系**：
  + 若$F_2(x_1, x_2) = F_1(\alpha x_1, \beta x_2)$，则$f_2(x_1, x_2) = \alpha\beta f_1(\alpha x_1, \beta x_2)$


### 根据已知分布$X, Y$求解分布$Z$的基本方法，及更高维的推广

+ 已知随机变量$X, Y$的分布，根据函数$Z = g(X, Y)$求解$Z$的随机分布：
  + **(离散, 离散) → 离散**：对$P\{Z = k\}$，计算所有可能对应的$X, Y$取值并求出对应概率，最后将这些概率相加
  + **(连续, 连续) → 连续**：
    + **分布函数定义法**：当随机变量$X, Y$**二者不独立**时优先考虑
      + $F_Z(z) = P\{g(x, y) \leqslant z\} = \iint_{g(x, y) \leqslant z}f(x, y){\rm d}x{\rm d}z$
      + $f_Z(z) = F'_Z(z)$
    + **卷积公式法**：
      + $Z = X + Y$，则$f_Z(z) = \int_{-\infty}^{\infty} f(x, z - x)\,{\rm d} x = \int_{-\infty}^{\infty} f(z - y, y)\,{\rm d} y$
        + 若$X, Y$独立，则$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x)\,{\rm d} x = \int_{-\infty}^{\infty} f_X(z - y) f_Y(y)\,{\rm d} y$
      + $Z = X - Y$，则$f_Z(z) = \int_{-\infty}^{\infty} f(x, x - z)\,{\rm d} x = \int_{-\infty}^{\infty} f(y + z, y)\,{\rm d} y$
        + 若$X, Y$独立，则$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(x - z)\,{\rm d} x = \int_{-\infty}^{\infty} f_X(y + z) f_Y(y)\,{\rm d} y$
      + $Z = XY$，则$f_Z(z) = \int_{-\infty}^{\infty} \frac{1}{|x|}f\left(x, \frac{z}{x}\right)\,{\rm d} x = \int_{-\infty}^{\infty} \frac{1}{|y|}f\left(\frac{z}{y}, y\right)\,{\rm d} y$
        + 若$X, Y$独立，则$f_Z(z) = \int_{-\infty}^{\infty} \frac{1}{|x|} f_X(x) f_Y\left(\frac{z}{x}\right)\,{\rm d} x = \int_{-\infty}^{\infty} \frac{1}{|y|} f_X\left(\frac{z}{y}\right) f_Y(y)\,{\rm d} y$
      + $Z = \frac{X}{Y}$，则$f_Z(z) = \int_{-\infty}^{\infty} |y|f(yz, y)\,{\rm d} y$
        + 若$X, Y$独立，则$f_Z(z) = \int_{-\infty}^{\infty} |y| f_X(yz) f_Y(y)\,{\rm d} y$
    + **最值函数（$\min, \max$）的分布**（此时$X, Y$独立）：
      + 对$Z = \max\{X, Y\}$，有$F_{\max}(z) = P\{\max\{X, Y\}\leqslant z\} = P\{X \leqslant z, Y \leqslant z\} = F(z, z)$，从而$$F_{\max}(z) = F_X(z) \cdot F_Y(z)$$
        + 推广：对$n$个独立随机变量，有$$F_{\max}(z) = \prod_{i = 1}^n F_{X_i}(z)$$
      + 对$Z = \min\{X, Y\}$，有$F_{\min}(z) = P\{\min\{X, Y\}\leqslant z\} = P\{(X \leqslant z)\cup(Y \leqslant z)\} = P\{X \leqslant z\} + P\{Y \leqslant z\} - P\{X \leqslant z, Y \leqslant z\} = F_X(z) + F_Y(z) - F(z, z)$，从而$$\begin{aligned}F_{\max}(z) =& F_X(z) + F_Y(z) - F_X(z) \cdot F_Y(z) \\ =& 1 - (1 - F_X(z))(1 - F_Y(z))\end{aligned}$$
        + 推广：对$n$个独立随机变量，有$$F_{\min}(z) = 1 - \prod_{i = 1}^n (1 - F_{X_i}(z))$$
  + **(离散, 连续) → 连续**：对离散变量展开使用**全概率公式**
+ 对**多维 → 多维**的分布求解，
  + 若**目标多维变量离散**且**可确定取值范围**，对每个取值分类讨论
  + 若**已知多维变量中存在离散分量**，在**该维变量**上用**全概率公式**展开

### $Y$的分布依赖于$X$的取值情况下的联合分布求解

+ 已知$X$的分布， **$Y$的分布依赖于$X$的取值**，求解$X, Y$的**联合分布**：
  1. 先**求解概率密度函数$f_{Y|X}(y|x)$**
  2. 再**根据$f(x, y) = f_X(x)f_{Y|X}(y|x)$**得到**联合概率密度函数**

## 数学期望、方差及数字特征相关

### 基本条件处理

+ 对连续随机变量$X$， **$P\{|X - E(X)| \leqslant \varepsilon\}$形式**的两种解法：
  + **需要确定下界**：套用**切比雪夫不等式**获得下界$P\{|X - E(X)| \leqslant \varepsilon\} \geqslant 1 - \frac{D(X)}{\varepsilon^2}$
  + **需要具体判断**：根据$X$的分布（如正态分布）来进行**直接判断**：特别是用于**两个**类似$P(|X - E(X)| \leqslant \varepsilon)$形式**进行比较**时
    + 设其概率密度函数为$f$，有$P\{|X - E(X)| \leqslant \varepsilon\} = \int_{E(X) - \varepsilon}^{E(X) + \varepsilon} f(x){\rm d}x$
+ 含**平均值**的期望 / 方差 / 协方差：
  1. 将式子尽可能地化简
  2. 将**平均值展开为$\frac{1}{n}\sum\limits_{i = 1}^n X_i$的形式**后求解

### 注意事项

+ 注意**区分**以下两个概念：
  + **协方差**：$\text{Cov}(X, Y) = E(XY) - E(X)E(Y)$
  + **相关系数**：$\rho = \frac{\text{Cov}(X, Y)}{\sqrt{DX}\sqrt{DY}}$
+ 注意**检查**：**对$\mathcal{N}(\mu, \sigma^2)$的变形，其$D(X)$表达式中大概率不含$\mu$** ！

### 不相关与独立的判定

+ **不相关**的判定：
  + $\text{Cov}(X, Y) = 0$
  + $E(XY) = E(X)E(Y)$
    + 二者完全等价
+ **独立**的判定：
  + 对**二维正态分布**、**0-1分布**：
    + **独立等价于不相关**
  + 对其他分布：
    + **离散分布**：
      + 求出**联合分布**和**每一维分布**的**具体**的**概率函数**
      + 检验式子 **$P\{X = x_i, Y = y_j\} = P\{X = x_i\}\cdot P\{Y = y_j\}$是否成立**
    + **连续分布**：
      + 求出**联合分布**和**每一维分布**的**具体**的**概率密度函数** / **分布函数**
      + 检验式子 **$f(x, y) = f_X(x)\cdot f_Y(y)$或$F(x, y) = F_X(x) \cdot F_Y(y)$是否成立**
    + 注意：**联合概率**（分布）必须用**不同于单个随机变量的、完全独立的方法求解**
+ 如果 **$X, Y$同分布**（不一定独立），**则$X + Y$和$X - Y$必定不相关**！
  + 证明：$\text{Cov}(X + Y, X - Y) = \text{Cov}(X, X) + \text{Cov}(Y, X) - \text{Cov}(X, Y) - \text{Cov}(Y, Y) = D(X) - D(Y) = 0$
+ **$\bar{X}$和$S^2$是相互独立的**！

### 数学期望与协方差的通用解法

+ **复杂函数**的**期望 / 协方差**的通用解法：如$E[(X + Y)^2]$、$\text{Cov}(X + Y, X - 2Y)$
  + **多项式展开法**：对于**独立同分布**变量构成的式子要优先考虑这一方法！
    + 如果式子为**多项式**，尝试将**式子展开**，如$E[(X + Y)^2] = E(X^2) + 2E(XY) + E(Y^2)$
      + 如果变量**相互独立 / 不相关**，则**乘法也可拆开**，如$E(XY) = E(X)E(Y)$
      + 如果变量**未说明相互独立 / 不相关**，则**只能拆加减法**，如$E(X \pm Y) = E(X) \pm E(Y)$
  + **完整换元法**：
    1. 将包含多个随机变量的式子化为**包含单一随机变量、且可以知晓该随机变量分布的式子**
    2. 根据新随机变量分布直接求解
       + 如代换$V = X + Y$，求出$V$的分布后再求解$E(V^2)$
  + **直接代入法**：
    + 直接**带入求解随机变量的分布公式**
      + 注意**概率密度函数**（如$f(x, y)$）**不一定是均匀分布**！

### 特殊的数学期望和协方差

+ 对**离散变量**$X, Y$，有
  + $E(X | Y) = \sum_{y_i} E(X | Y = y_i)P\{Y = y_i\}$
+ 对**任意变量**$X, Y$，有
  + $\text{Cov}(X, Y) = E(XY) - E(X)E(Y)$
    + 这里$XY$**就是$X$和$Y$的乘积！** 不是~~古典概型中所谓二者同时发生的概念~~
+ **已知方差求$E(X^2)$** ：
  + $E(X^2) = D(X) + (EX)^2$
+ 求解$E(g(X))$：这里$g(X)$可以为**任意函数**
  + 已知概率密度函数$f_X(x)$，则$E[g(X)] = \int_{-\infty}^{+\infty} g(x)f_X(x)\,{\rm d}x$
+ 求解$E(h(X, Y))$：这里 **$h(X, Y)$可以为任意函数**（如$\max\{X, Y\}$）
  + 已知联合概率密度函数$f(x, y)$ ，则$E[h(X, Y)] = \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} h(x, y)f(x, y)\,{\rm d}x{\rm d}y$
+ 求解$E(|X|)$：
  1. 若$X$的概率密度函数为$f(x)$，则
    + $|X|$的**概率密度函数**为$f_{|X|}(x) = \begin{cases} 0, & x \leqslant 0 \\ f(x) + f(-x) , & x > 0 \end{cases}$
    + $|X|$的**分布函数**为$F_{|X|}(x) = \begin{cases} 0, & x \leqslant 0 \\ \int_{-x}^{x} f(t)\,{\rm d}t, & x > 0 \end{cases}$
  2. **采用$E(|X|) = \int_{0}^{+\infty} x (f(x) + f(-x))\,{\rm d}x$求解**

### 方差

+ 方差的数学性质：
  + $D(nX) = n^2 D(X)$
  + $D\left(\frac{X}{n}\right) = \frac{1}{n^2}D(X)$
  + $D(\bar{X}) = \frac{1}{n}D(X)$
  + 若$E(X) = E(Y)$，则$D(aX + bY) = a^2 D(X) + b^2 D(Y)$
+ **复杂函数的方差**：如$D[(X + Y)^2]$
  + 一律**采用$D(X) = E(X^2) - (EX)^2$的求法**！
+ **离散变量**的$D(X | Y)$求解：
  + **全概率公式法**：$D(X | Y) = \sum_{y_i} D(X | Y = y_i)P\{Y = y_i\}$
  + **期望法**：$D(X | Y) = E(X^2 | Y) - E(X | Y)^2$
+ 求解$D(|X|)$：
  + 采用$D(|X|) = E(X^2) - E(|X|)^2$求解
    + $E(X^2)$可以直接使用$E(X^2) = \int_{-\infty}^{+\infty} x^2 f(x)\,{\rm d}x$求解
    + $E(X^2)$也可以通过$E(X^2) = D(X) + E(X)^2$求解

### $Y$的分布依赖于$X$的取值情况下的期望和方差

+ 对任意两个随机变量$X, Y$，若 **$Y$的分布与$X$的取值相关** ，必有：
  + $Y$的**期望$E(Y) = E(E(Y|X))$**
    + 翻译：若$Y$的分布依赖于$X$的取值，则在$X$、$Y$均自由任意取值的情况下，$Y$的期望即为$E(Y|X = x)$这一由$X$派生出期望的期望
    + 证明：
      + 注意到$$\begin{aligned}E(Y) =& \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} yf(x, y)\,{\rm d}x{\rm d}y \\ =& \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} yf(y | x)f_X(x)\,{\rm d}x{\rm d}y \\ =& \int_{-\infty}^{+\infty}f_X(x) \,{\rm d}x\int_{-\infty}^{+\infty} yf(y | x) \,{\rm d}y \\ =& \int_{-\infty}^{+\infty}f_X(x) E(Y|X = x) \,{\rm d}x \end{aligned}$$同时有$E(E(Y|X)) = \int_{-\infty}^{+\infty} f_X(x)E(Y|X = x)\,{\rm d}x$，从而$E(Y) = E(E(Y|X))$
    + 使用：计算$E(Y)$时，
      1. **计算$E(Y|X = x)$作为$g(X)$** ；其中$g(X)$为一个和$x$相关的函数，此处$x$与$X$可互相替换
      2. 计算 **$g(x)$的期望$E(g(X))$** ，即$\int_{-\infty}^{+\infty} g(x) f_X(x)\,{\rm d}x$
      3. 最终的$E(Y)$即为$E(g(X))$
  + $Y$的**方差$D(Y) = E(D(Y|X)) + D(E(Y|X))$**
    + 使用：计算$D(Y)$时，
      1. **计算$E(Y|X = x)$作为$g(X)$** ；其中$g(X)$为一个和$x$相关的函数，此处$x$与$X$可互相替换
      2. **计算$g(x)$的方差$D(g(X))$**
      3. **计算$D(Y|X = x)$作为$h(X)$** ；其中$h(X)$为一个和$x$相关的函数，此处$x$与$X$可互相替换
      4. **计算$h(x)$的期望$E(h(X))$**
      5. 最后 **$Y$的方差$D(Y) = D(g(X)) + E(h(X))$**


### 独立同分布与数学期望、方差、协方差

+ **独立同分布**的变量的期望 / 方差 / 协方差：
  1. 将 **$\bar{X}$拆成$\frac{1}{n}\sum\limits_{j = 1}^n X_j$** ；$D(X)$等同理
  2. 利用 **$X_i$和$X_j$互相独立**进行整理
  3. 利用**同分布**进行化简
     + 可利用结论：对于**独立同分布**的变量，由于$\text{Cov}(X_i, X_j) = 0, i \neq j$，从而**有$\text{Cov}(X, \bar{X}) = \frac{1}{n}D(X_i)$**
+ 对**独立同分布的$X_i, i = 1, 2, \cdots, n$** ，其**样本方差的期望$E(S^2)$** 满足$E(S^2) = D(X)$
  + 证明：$S^2 = \frac{1}{n - 1} \sum\limits_{i = 1}^n (X_i - \bar{X})^2 = \frac{1}{n - 1} \sum\limits_{i = 1}^n (X_i^2 - 2X_i\bar{X} + \bar{X}^2)$，由于独立同分布可知$E(X_iX_j) = E(X_i)E(X_j) = E(X)^2$，从而有$$\begin{aligned} E(S^2) =& \frac{1}{n - 1} \sum\limits_{i = 1}^n \left(E(X_i^2) - 2E(X_i\bar{X}) + E(\bar{X}^2)\right) \\ =& \frac{1}{n - 1} \sum\limits_{i = 1}^n \left(E(X_i^2) - 2E(X_i \cdot \frac{1}{n}\sum\limits_{j = 1}^nX_j) + E(\bar{X})^2 + D(\bar{X})\right) \\ =& \frac{1}{n - 1} \sum\limits_{i = 1}^n \left(E(X_i^2) - \frac{2}{n}E(X_i^2) - \frac{2}{n} \cdot (n - 1)E(X)^2 + E(X)^2 + \frac{1}{n}D(X)\right) \\ =& \frac{1}{n - 1} \sum\limits_{i = 1}^n \left(\frac{n - 2}{n}E(X^2) - \frac{n - 2}{n}E(X)^2 + \frac{1}{n}D(X)\right) \\ =& \frac{n}{n - 1} \cdot \left(\frac{n - 2}{n} + \frac{1}{n}\right) D(X) \\ =& D(X) \end{aligned}$$
  + 注意**区别**：$E(S^2) = D(X)$，但 **$D(X)$的矩估计量为$\frac{n - 1}{n}S^2$**

## 大数定律及中心极限定理

### 基本条件处理

+ 对**连续随机变量$X$** （可以为数学期望、方差等统计量），设其概率密度函数为$f$，有$P\{|X - \mu| < \varepsilon\} = \int_{\mu - \varepsilon}^{\mu + \varepsilon} f(x){\rm d}x$
  + 在此基础上，**若$f$的表达式中含参数$n$** ，可以**考查$n \to \infty$时是否满足**$P \to 1$
  + 这一方法既可以用于考察**大数定律 / 中心极限定理**，也用于**相合性的检验**
+ 大数定律的条件：
  + **辛钦大数定律**：**独立同分布 + 数学期望存在**
    + 注意：**服从同一离散型 / 连续型分布不能保证数学期望存在**！
  + **切比雪夫大数定律**：**两两不相关 + 存在常数$c$使得$D(X_i) \leqslant c$**

### 注意事项

+ 只要遇到**概率的极限**（形如$\overset{P}{\longrightarrow}$、$\lim\limits_{n \to \infty}P = $）以及**独立同分布随机变量的和**$\sum\limits_{i = 1}^n X_i$：
  + 第一时间考虑**大数定律 / 中心极限定理**
+ 只要遇到 **“近似服从”** 这类词汇：
  + 第一时间考虑**中心极限定理**
  + 只需要求出原分布$X$的期望$E(X)$和方差$D(X)$，则$\bar{X} = \frac{1}{n}\sum\limits_{i = 1}^n X_i$按照中心极限定理，应当**近似服从正态分布$\mathcal{N}(E(X), D(X))$**


## 统计量及其分布

### 基本条件处理

+ 卡方分布推论：对**服从标准正态分布$\mathcal{N}(0, 1)$的$X$** ，有$E(X^2) = E(\chi^2(1)) = 1, D(X^2) = D(\chi^2(1)) = 2$

### 注意事项

+ $F(m, n)$表示**分子为$\frac{\chi^2(m)}{m}$，分母为$\frac{\chi^2(n)}{n}$** ，而不是反过来！
+ 对于$t$分布和$F$分布，实际操作时，只要**组成该分布的正态分布的期望$\mu = 0$** ，则**方差可以为任意的$\sigma^2$** （在相除时会消掉）

## 参数统计

### 基本条件处理

+ 如果**最大似然函数$L$表达式和$X_i$无关**，或 **$L$恒定为常数**：例如$X$服从均匀分布
  + 最大似然函数不需要求对数，**直接考察**使其**能够正确存在**的**参数范围**！

### 估计量求解的通用方法

+ 求解参数$\theta$的**矩估计量**：已知$X$的概率密度函数$f(x; \theta)$；未说明默认一阶，二阶同理
  1. 先**求解出$E(X) = \int_{-\infty}^{\infty} xf(x; \theta) \,{\rm d}x = g(\theta)$** 为关于待定概率参数$\theta$的函数
  2. 然后**令$g(\theta) = \bar{X}$得到$\hat{\theta} = h(\bar{X})$** ，则$h(\bar{X})$为$\theta$的矩估计量
  3. 如果**一阶原点矩为$0$ / 常数$C$（即和参数$\theta$无关）**：
     + 依次尝试更换**二阶原点矩、一阶中心距、二阶中心距**等
+ 求解参数$\theta$的**最大似然估计量**：已知$X$的概率密度函数$f(x; \theta)$
  1. **写出似然函数$L(\theta) = \prod\limits_{i = 1}^n f(x = X_i; \theta)$**
  2. 将$L$中**底数相同的归到一起**，剩余的**有规律的乘积式归到一起**
  3. 求出**对数似然函数$\ln L(\theta)$**
  4. 此时先前各类可以快速写成若干个$\ln$相关的式子；将**含$\theta$的归到一起**，**剩余“常量”项归到一起**
  5. **求出$\frac{\partial \ln L(\theta)}{\partial \theta}$** ，此时“常量”项直接舍去
  6. 讨论$\frac{\partial \ln L(\theta)}{\partial \theta}$的取值：
     + 若 **$\frac{\partial \ln L(\theta)}{\partial \theta}$恒为$0$ / 恒大于$0$ / 恒小于$0$** ，即$\frac{\partial \ln L(\theta)}{\partial \theta} \equiv 0$或$\frac{\partial \ln L(\theta)}{\partial \theta} \gtrless 0$，则**考察**使 **$\theta$能够正确存在**的**参数范围**，取**这个范围中使得$L(\theta)$最大的点（通常为端点）**
     + 若 **$\frac{\partial \ln L(\theta)}{\partial \theta}$在某个点为$0$** ，则**将方程$\frac{\partial \ln L(\theta)}{\partial \theta} = 0$变形为$\theta = F(X_i, 1 \leqslant i \leqslant n)$的形式**，记此时的$\theta$为$\hat{\theta}$
  7. 观察$\hat{\theta}$表达式，**能否用更简洁的表达替换**
     + 例如$\frac{1}{n}\sum\limits_{i = 1}^n X_i$可替换为$\bar{X}$
+ 求解某个**统计量、概率或表达式**的**矩估计量 / 最大似然估计量**：如$D(X)$、$P\{e^X > 1\}$
  1. 先用矩估计 / 最大似然估计**求出**概率密度函数$f(x; \theta)$中的**未知参数$\theta$的估计$\hat{\theta}$**
  2. 再写出**该统计量、概率或表达式用$\theta$表示的式子$g(\theta)$**
     + 如$D(X) = \frac{\theta^2}{12}$、$P\{e^X > 1\} = e^{-\theta}$
  3. **检查$g(\theta)$是否是关于$\theta$的单调函数**
     + 一般都是，但**必须要写出来**：**根据“矩估计 / 最大似然估计的不变性原理”** ！
  4. 将估计$\hat{\theta}$的表达式代入$g(\theta)$，得到最终估计的表达式

### 无偏性、相合性的检验

+ 对某个估计$\hat{\theta}$判断**是否无偏**：
  1. **将$\hat{\theta}$表达式中$h(X_i)$部分替换为对应的$E(h(X))$** ，得到$E(\hat{\theta})$
     + 可能有多个不同的$h(X_i)$，需**一一对应替换**
     + 例子：$\sum\limits_{i = 1}^n X_i \to nE(X)$、$\sum\limits_{i = 1}^n X_i^2 \to nE(X^2)$
  2. 根据$X$的**分布函数 / 概率密度函数**计算**每一个$E(h(X))$**
  3. 将**结果代入$E(\hat{\theta})$，检查其是否等于$\theta$**
+ 对某个估计$\hat{\theta}$判断**是否相合**（一种方法）：
  1. 对$\hat{\theta}$表达式，依次计算$E(\hat{\theta})$和$D(\hat{\theta}) = E(\hat{\theta}^2) - E(\hat{\theta})^2$（此处这二者应当与样本个数$n$相关）
  2. 使用**切比雪夫不等式**，有$P\{|\hat{\theta} - E(\hat{\theta})| \geqslant \varepsilon\} \leqslant \frac{D(\hat{\theta})}{\varepsilon^2}$
  3. **证明$\lim\limits_{n \to \infty} E(\hat{\theta}) = \theta$且$\lim\limits_{n \to \infty} \frac{D(\hat{\theta})}{\varepsilon^2} \to 0$**
  4. 代入上式，可解得$\lim\limits_{n \to \infty} P\{|\hat{\theta} - E(\hat{\theta})| \geqslant \varepsilon\} = \lim\limits_{n \to \infty} P\{|\hat{\theta} - \theta| \geqslant \varepsilon\} \leqslant \lim\limits_{n \to \infty} \frac{D(\hat{\theta})}{\varepsilon^2} = 0$

## 假设检验

### 基本条件处理

+ 对在某个区间上**均匀分布**的、概率密度函数为$p$随机变量$X$ ，则各个样本的概率密度函数就是$p$
  + 例子：该均匀分布在$[a, b]$上的概率密度是$\frac{1}{b - a}$，则各个样本的概率密度函数就是$\frac{1}{b - a}$
  + 注意：均匀分布时，各个样本的概率密度函数**和样本取值无关**！
+ 上一条的拓展：对**在各个互不重合的区间上均匀分布的随机变量$X$** ，如果已知各个样本的具体值，则**各个样本的概率密度**为**其所在区间的概率密度**
  + 例子：在$[a_1, b_1], [a_2, b_2], \cdots, [a_n, b_n]$上的概率密度分别是$p_1, p_2, \cdots, p_n$，且$X \in [a_1, b_1]$，则$X$的概率密度函数为$p_1$

### 注意事项

+ 对于题目给出参数的情况（如$\varPhi(1.645) = 0.95$），代入计算时除非题目要求，否则**不需要自己四舍五入**！
+ 记忆时**不要混淆**：$z_{\frac{\alpha}{2}}$为**上$\frac{\alpha}{2}$分位点**，即$P\{X > z_{\frac{\alpha}{2}}\} = \frac{\alpha}{2}$，而**不是**~~分布函数为$\frac{\alpha}{2}$的点~~
+ 记忆时**不要混淆**：
  + **置信水平**为$1 - \alpha$（可理解为 **“检验问题不明显的区间的占比”、“没有问题的区间的占比”**）
  + **显著性水平**为$\alpha$（可理解为 **“检验问题明显的区间的占比”、“存在问题的区间的占比”**）
  + 常用的**检验区间端点**是**上$\frac{\alpha}{2}$分位点**
+ 如果**题目问“检验某个××是否成立”**，其对应的是**备择假设$H_1$** ，而**不是**~~原假设$H_0$~~
  + 理解：在现实应用中，备择假设是“出现问题”的假设，而题目要检验的一般都是“出现问题的可能”
+ **等号（如$\geqslant$、$\leqslant$）一般是在原假设$H_0$里**
  + 例：$H_0: \mu_1 \leqslant \mu_2, H_1: \mu_1 > \mu_2$是**合理**的，而$H_0: \mu_1 < \mu_2, H_1: \mu_1 \geqslant \mu_2$是**不合理**的


### 置信区间的计算、期望

+ 求解**随机变量$Y = g(X)$关于$\theta$的置信区间** （其中$X$服从概率密度函数为$f(x; \theta)$的分布，$\theta$为未知参数；$g$可以是**函数**，也可以是$E(e^X)$等**关系式**）：
  1. 根据$f(x; \theta)$的分布，求出 **$X$中$\theta$的置信区间$(a, b)$**
  2. 计算$Y = g(X) = h(\theta)$
  3. 检验一下$h$关于$\theta$的单调性
     + 一般都是，但**必须要写出来**：**根据“矩估计 / 最大似然估计的不变性原理”** ！
  4. $Y$的置信区间为$(h(a), h(b))$
+ 求解$X$显著性水平为$\alpha$的**置信区间上下限、长度等参量**的**期望**
  1. **写出$X$的置信区间以及相关参量（如长度）的表达式**；如对正态分布$\mathcal{N}(\mu, \sigma^2)$（二者均未知），关于$\mu$的显著性水平为$\alpha$的置信区间为$\left(\bar{X} - \frac{S}{\sqrt{n}} t_{\frac{\alpha}{2}}(n - 1), \bar{X} + \frac{S}{\sqrt{n}} t_{\frac{\alpha}{2}}(n - 1)\right)$，置信区间长度为$\frac{2S}{\sqrt{n}} t_{\frac{\alpha}{2}}(n - 1)$
  2. 对每个表达式，将**其中的未知参量 / 随机变量**进行**期望求解**；如$\bar{X} - \frac{S}{\sqrt{n}} t_{\frac{\alpha}{2}}(n - 1)$中需要求解$E(\bar{X}) = E(X) = \mu$、$E(S) = \sigma$，注意此处$t_{\frac{\alpha}{2}}(n - 1)$是常数，无需求解期望
  3.  将**结果代入原来的每个表达式**，得到所求期望

### 置信水平、显著性水平等的表述翻译

+ 题目提供**正态分布函数**的条件翻译：
  + $\Phi(a) = b$ $\Longleftrightarrow$ 显著性水平$\alpha = 2(1 - b)$时，$z_{1 - b} = a$
    + 例子：$\Phi(1.96) = 0.975$ $\Longleftrightarrow$ $z_{0.025} = z_{\frac{0.05}{2}} = 1.96$，对应显著性水平$\alpha = 2(1 - 0.975) = 0.05$
+ **抽样检测**的问题中，**原假设（$\mu = a$）** 的表述：
  + 一般是“若某件物品合格，则抽样**合格的概率为$1 - \alpha$**”，其中$\alpha$为**显著性水平**
