## 基本处理方法

### 重要结论

+ **$\bm{B}$的列向量均不构成$\bm{A}\bm{x} = \bm{0}$的解**：
  + $\bm{A}\bm{B} \neq \bm{O}$
+ 对**列向量组**$\bm{\alpha}_1, \bm{\alpha}_2, \cdots, \bm{\alpha}_n$，令$\bm{A}$为各个向量构成的矩阵，则有$$k_1\bm{\alpha}_1 + k_2\bm{\alpha}_2 + \cdots + k_n\bm{\alpha}_n = \bm{A}\begin{bmatrix} k_1 \\ k_2 \\ \vdots \\ k_n \end{bmatrix}$$

### 处理方法
+ 对$\bm{B} = [k_{11}\bm{\alpha}_1 + \cdots + k_{1n}\bm{\alpha}_n, \cdots, k_{n1}\bm{\alpha}_1 + \cdots + k_{nn}\bm{\alpha}_n]$形式的矩阵处理：
  + **构造矩阵$\bm{K} = \begin{bmatrix} k_{11} & \cdots & k_{n1} \\ \vdots & \ddots & \vdots \\ k_{1n} & \cdots & k_{nn} \end{bmatrix}$** ，从而$\bm{B} = \bm{A}\bm{K}$，其中$\bm{A} = [\bm{\alpha}_1, \cdots, \bm{\alpha}_n]$
    + 例如$\bm{B} = [\bm{\alpha}_1 + 2\bm{\alpha}_2, 3\bm{\alpha}_1 + 4\bm{\alpha}_2]$，取$\bm{A} = [\bm{\alpha}_1, \bm{\alpha}_2]$，构造$\bm{K} = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$，从而$$\bm{B} = [\bm{\alpha}_1, \bm{\alpha}_2]\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix} = \bm{A}\bm{K}$$
+ 对于**只包含若干个平方的方程组**的一种解法：
  + **对每个平方写出放缩范围**，然后**找出两个放缩范围所夹逼得到的取值**
    + 如$\begin{cases} a^2 + b^2 = 1 \\ (a + 2)^2 + c^2 = 1 \end{cases}$，有$\begin{cases} a^2 \leqslant 1 \\ (a + 2)^2 \leqslant 1 \end{cases}$，解得$\begin{cases} -1 \leqslant a \leqslant 1 \\ -3 \leqslant a \leqslant -1 \end{cases}$，从而$a = -1$，进而有$b = 0, c= 0$


### 抽象向量和向量表达式组合的求解

+ 给出了 **$n$个$n$维抽象向量**（即没有具体形式的向量，如$\bm{\alpha}_1$、$\bm{A}\bm{\alpha}_1$、$\bm{\alpha}_1 + \bm{\alpha}_2$；此处表记为$\bm{\beta}_1, \bm{\beta}_2, \cdots, \bm{\beta_n}$）以及**和这些向量相关的关系式**：
  1. **证明线性无关**：
     + 一般而言，需要证明$\bm{\beta}_1, \bm{\beta}_2, \cdots, \bm{\beta_n}$**线性无关**
  2. **构造矩阵**：
     + 将其作为列向量**构造为一个矩阵**：$\bm{B} = [\bm{\beta}_1, \bm{\beta}_2, \cdots, \bm{\beta_n}]$
  3. **计算参数矩阵**：
     + **计算$\bm{A}[\bm{\beta}_1, \bm{\beta}_2, \cdots, \bm{\beta_n}]$** ，其中$\bm{A}$通常为**和向量表达式内某个参数有关的矩阵**
  4. **整理得到结果$\bm{C}$** ：
     + 整理上式，一般在这里可以**根据题目给出的关系式得到$\bm{A}[\bm{\beta}_1, \bm{\beta}_2, \cdots, \bm{\beta_n}] = [\bm{\beta}_1, \bm{\beta}_2, \cdots, \bm{\beta_n}]\bm{C}$** ，其中$\bm{C}$为**能知晓每个元素具体形式**的矩阵
  5. **利用$\bm{A}$、$\bm{C}$相似**：
     + 由于$\bm{\beta}_1, \bm{\beta}_2, \cdots, \bm{\beta_n}$线性无关，**可知$\bm{B}$可逆**，从而$\bm{B}^{-1}\bm{A}\bm{B} = \bm{C}$， **$\bm{A}$和$\bm{C}$相似**
  6. **求解特征值**：
     + **求解$\bm{C}$的特征值$\lambda_i$，即为$\bm{A}$的特征值**
  7. **求解$\bm{C}$的特征向量**：
     + **求解$\bm{C}$的特征值$\lambda_i$对应的特征向量$\xi_i$**
  8. **计算$\bm{A}$的特征向量**：
     + 由$\bm{A} = \bm{B}\bm{C}\bm{B}^{-1}$知 **$\bm{B}\xi_i$为$\bm{A}$关于特征值$\lambda_i$的特征向量**

## 行列式

### 基本条件处理

+ 对**某一行 / 列绝大部分为$0$** 的行列式，**使用该行 / 列展开求解**
+ 遇到**第一行 / 第一列全部为$1$** 的行列式，优先考虑**范德蒙行列式**
+ 在化简高阶行列式时**不一定要化为完全对角线的形式**：
  1. 化简使得**对角线位置由若干个二阶 / 三阶子矩阵构成**，形如$\begin{bmatrix} \bm{A} & \bm{O} \\ \bm{C} & \bm{B} \end{bmatrix}$
  2. **使用$\begin{vmatrix} \bm{A} & \bm{O} \\ \bm{C} & \bm{B} \end{vmatrix} = |\bm{A}||\bm{B}|$** 进行求解
+ 对**对角矩阵**和**副对角线的对角矩阵**，**非对角线 / 副对角线对应元素的余子式均为$0$** ！


### 注意事项

+ **求解行列式前**，尽量**先化简行列式**！
+ 一定要区分使用**主对角线（左上-右下）** 的行列式和使用**副对角线（右上-左下）** 的行列式！
  + 主对角线为$a_1, \cdots, a_n$：行列式为$\prod\limits_{i = 1}^n a_i$
  + 副对角线为$a_1, \cdots, a_n$：行列式为$(-1)^{\frac{n(n - 1)}{2}}\prod\limits_{i = 1}^n a_i$
    + 对于$n$阶副对角线行列式： **$-1$的指数不唯一**，但**该指数的奇偶性必定相同**！（即$(-1)^3 = (-1)^5 = -1$）


### 加边法的应用

+ 一种**加边法**的应用场景：**每行 / 列对应的项绝大部分都有共同因子**，也就是可以添加一行后通过加乘将这些项消去
  + 常见例子：
    + $\begin{vmatrix} k + \lambda_1 & k & k \\ k & k + \lambda_2 & k \\ k & k & k + \lambda_3 \end{vmatrix} = \begin{vmatrix} 1 & k & k & k \\ 0 & k + \lambda_1 & k & k \\ 0 & k & k + \lambda_2 & k \\ 0 & k & k & k + \lambda_3 \end{vmatrix} = \begin{vmatrix} 1 & k & k & k \\ -1 & \lambda_1 & 0 & 0 \\ -1 & 0 & \lambda_2 & 0 \\ -1 & 0 & 0 & \lambda_3 \end{vmatrix}$
    + $\begin{vmatrix} a_1 + \lambda_1 & a_2 & a_3 \\ a_1 & a_2 + \lambda_2 & a_3 \\ a_1 & a_2 & a_3 + \lambda_3 \end{vmatrix} = \begin{vmatrix} 1 & a_1 & a_2 & a_3 \\ 0 & a_1 + \lambda_1 & a_2 & a_3 \\ 0 & a_1 & a_2 + \lambda_2 & a_3 \\ 0 & a_1 & a_2 & a_3 + \lambda_3 \end{vmatrix} = \begin{vmatrix} 1 & a_1 & a_2 & a_3 \\ -1 & \lambda_1 & 0 & 0 \\ -1 & 0 & \lambda_2 & 0 \\ -1 & 0 & 0 & \lambda_3 \end{vmatrix}$
    + $\begin{vmatrix} a_1b_1 + \lambda_1 & a_2b_1 & a_3b_1 \\ a_1b_2 & a_2b_2 + \lambda_2 & a_3b_2 \\ a_1b_3 & a_2b_3 & a_3b_3 + \lambda_3 \end{vmatrix} = \begin{vmatrix} 1 & a_1 & a_2 & a_3 \\ 0 & a_1b_1 + \lambda_1 & a_2b_1 & a_3b_1 \\ 0 & a_1b_2 & a_2b_2 + \lambda_2 & a_3b_2 \\ 0 & a_1b_3 & a_2b_3 & a_3b_3 + \lambda_3 \end{vmatrix} = \begin{vmatrix} 1 & a_1 & a_2 & a_3 \\ -b_1 & \lambda_1 & 0 & 0 \\ -b_2 & 0 & \lambda_2 & 0 \\ -b_3 & 0 & 0 & \lambda_3 \end{vmatrix}$

### 余子式与代数余子式的计算

+ $\sum\limits_{j = 1}^n A_{ij} = \begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & & \vdots \\ 1 & 1 & \cdots & 1 \\ \vdots & \vdots & & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \\ \end{vmatrix}$，其中全为$1$的行在第$i$行
  + 遇到类似于**求解所有代数余子式的和**：$\sum\limits_{i = 1}^n \sum\limits_{j = 1}^n A_{ij}$时，考虑此方法（即将**各行 / 各列分别依次替换为$1$后**进行求和计算）
+ **同一行 / 列的若干（代数）余子式的线性组合**，例如$\sum\limits_{j = 1}^n k_jA_{ij}$：
  + 考虑**将原行列式的该行 / 列替换为对应的求解式中（代数）余子式的系数**，如$$\sum\limits_{j = 1}^n k_j A_{ij} = \begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & & \vdots \\ k_1 & k_2 & \cdots & k_n \\ \vdots & \vdots & & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \\ \end{vmatrix}$$其中系数$k_j$在第$i$行

### 分块矩阵的行列式

+ 计算**分块矩阵$\bm{X} = \begin{bmatrix} \bm{A} & \bm{B} \\ \bm{C} & \bm{D} \end{bmatrix}$（其中$\bm{A}$可逆；$\bm{B}， \bm{C}$不一定为方阵）的行列式**——**结论**：$|\bm{X}| = |\bm{A}||\bm{D} - \bm{C}\bm{A}^{-1}\bm{B}|$
  1. **不写入过程**：令$\bm{P} = \begin{bmatrix} \bm{E}_1 & \bm{O} \\ \bm{M} & \bm{E}_2 \end{bmatrix}$，其中$\bm{Y}$每个分块和$\bm{X}$每个分块形状相同，且$\bm{E}_1$、$\bm{E}_2$为单位矩阵
  2. **不写入过程**：计算$\bm{P}\bm{X} = \begin{bmatrix} \bm{A} & \bm{B} \\ \bm{M}\bm{A} + \bm{C} & \bm{M}\bm{B} + \bm{D} \end{bmatrix}$
  3. **不写入过程**：令$\bm{M}\bm{A} + \bm{C} = \bm{O}$，得到$\bm{M} = -\bm{C}\bm{A}^{-1}$
  4. 在答题纸上写出此时的$\bm{P} = \begin{bmatrix} \bm{E}_1 & -\bm{C}\bm{A}^{-1} \\ \bm{O} & \bm{E}_2 \end{bmatrix}$，并**标明$|\bm{P}| = 1$**
  5. 从而**此时有$|\bm{X}| = |\bm{P}\bm{X}| = \begin{vmatrix} \bm{A} & \bm{B} \\ \bm{O} & -\bm{C}\bm{A}^{-1}\bm{B} + \bm{D} \end{vmatrix} = |\bm{A}||\bm{D} - \bm{C}\bm{A}^{-1}\bm{B}|$**
+ 一种应用场景：对于**类似于$\bm{X} = \begin{bmatrix} \bm{A} & \bm{B} \\ \bm{C} & \bm{D} \end{bmatrix}$的抽象分块矩阵，找到让其可逆的条件**，此时$|\bm{X}| \neq 0$可用于简化求解式子

## 矩阵基础性质与矩阵运算

### 基本条件处理

+ 对于**多个数列之间存在线性的递推关系**（如$a_n = 2a_{n - 1} + 3b_{n - 1}, b_n = 3a_{n - 1} + 2b_{n - 1}$），可以**用构造矩阵转化为递推式**，然后用**矩阵的幂**求解
+ 若$\bm{C}\bm{A} = \bm{C}\bm{B}$且$\bm{C}$可逆，则$\bm{A} = \bm{B}$
+ 已知某个条件（如$\bm{A}^T\bm{A} = \bm{O}$），要**判断$\bm{A} = \bm{O}$** ：
  + 通过变形，**得到$r(\bm{A}) = 0$**
+ 对于**分块矩阵**，求逆时可以**尝试分块求解**


### 注意事项

+ 判断某个表达式（如$\bm{A}\bm{A}^\text{*}$）是否是**对称矩阵**：
  + **用定义$\bm{A}^T = \bm{A}$！**
+ $\bm{A} - \bm{A}^T$**不是**~~对称矩阵~~，而是**反对称矩阵**（$\bm{M} = -\bm{M}^T$）！

### 初等矩阵

+ 初等矩阵$E_{ij}(k)$是对应**第$j$行乘$k$加到第$i$行 / 第$i$列乘$k$加到第$i$列**，二者顺序不要搞混、搞错！
+ 对于初等矩阵，**行变换乘在左，列变换乘在右**！
+ 对于**复合矩阵幂**的式子（如$\bm{A}^5\bm{B}\bm{C}^4$）：
  1. 注意观察**带幂次的矩阵**是否是**初等矩阵**（一般都是）
  2. 如果是，直接用**初等矩阵的定义**快速求解
    + 再次**注意**：
      + 初等矩阵**行变换乘在左，列变换乘在右**
      + 同时**不能一次性多次操作，要在每降低某项的一个幂次时都操作一次**

### 矩阵的幂

+ **求解$\bm{A}^n$** ，依次尝试以下方法：
  1. $r(\bm{A}) = 1$：
     + 令$\bm{A} = \bm{x}\bm{x}^T$，则$\bm{A}^n = \bm{x}(\bm{x}^T\bm{x})^{n - 1}\bm{x}^T = (\bm{x}^T\bm{x})^{n - 1}\bm{A} = \text{tr}(\bm{A})^{n - 1}\bm{A}$
  2. $\bm{A}$为**上三角 / 下三角矩阵**：
     1. 拆分为对角矩阵$\bm{C}$和对角线为$0$的上三角 / 下三角矩阵$\bm{B}$的和
     2. 注意到对$n$阶方阵，$\bm{B}^n = \bm{O}$
     3. 从而对$\bm{A}^n = (\bm{C} + \bm{B})^n$进行二项式展开，即可得到答案
  3. 对任意矩阵：
     1. 计算$\bm{A}$的**特征值构成的矩阵$\bm{\Lambda}$**和**对应特征向量构成的矩阵$\bm{P}$**
        + 这里$\bm{\Lambda}$**可以不满秩**，不影响之后的计算！
     2. **根据$\bm{P}$计算$\bm{P}^{-1}$**
        + **实对称矩阵**可直接**对特征向量单位化**，则$\bm{P}^{-1} = \bm{P}^T$！
     3. $\bm{A}^n = \bm{P}\bm{\Lambda}^n\bm{P}^{-1}$

### 可逆判断

+ 判断**可逆**：
  + 求解**行列式$|\bm{A}| \neq 0$**
  + 对 **$\bm{A} + k\bm{E}$形式**：
    + 一种**常见构造**是$(\bm{A} + k\bm{E})(\bm{B} + l\bm{E}) = g\bm{E}$，具体系数可以根据题设配出
  + 对$\bm{A}$**存在表达式**的情况：
    + 尝试**计算$\bm{A}^2$**并尝试**整理出$\bm{A}^2 = a\bm{A} + b\bm{E}$的形式**，若该形式存在则$\bm{A}^{-1} = \frac{1}{b}(\bm{A} - a)$
+ 判断**不可逆**：
  + 求解**行列式$|\bm{A}| = 0$**
  + 若 **$\bm{A}$存在表达式**，可以构造方程$\bm{A}\bm{x} = \bm{0}$并尝试**找到$\bm{x}$的表达式**


### 矩阵乘法的错误结论

+ $\bm{A}^2 = \bm{E}$**无法推出**$\bm{A} = \pm\bm{E}$
  + 反例：$\bm{A} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$

+ $\bm{A}^2 = \bm{O}$**无法推出**$\bm{A} = \bm{O}$
  + 反例：$\bm{A} = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$

+ $\bm{A}^2 = \bm{A}, \bm{A} \neq \bm{O}$**无法推出**$\bm{A} = \bm{E}$
  + 反例：$\bm{A} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$

+ $\bm{A}\bm{B} = \bm{O}$**无法推出**$\bm{A} = \bm{O}$或$\bm{B} = \bm{O}$
  + 反例：$\bm{A} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \bm{B} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$

+ $\bm{A} \neq \bm{O}, \bm{B} \neq \bm{O}$**无法推出**$\bm{A}\bm{B} \neq \bm{O}$
  + 反例（同上）：$\bm{A} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \bm{B} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}$

+ $\bm{A}\bm{B} = \bm{B}$**无法推出**$\bm{A} = \bm{E}$
  + 反例：$\bm{A} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \bm{B} = \begin{bmatrix} 2 & 0 \\ 0 & 0 \end{bmatrix}$

+ $\bm{A}_{m \times n}$（$m < n$）可以推出$r(\bm{A}^T\bm{A}) < n$，但$r(\bm{A}^T\bm{A}) < n$**无法推出**$\bm{A}$必然为$m \times n$（$m < n$）矩阵
  + 反例：$\bm{A} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$（即任何不可逆方阵）


### 矩阵多项式

+ 已知**某个矩阵可逆 / 若干个矩阵的和或积是$\bm{E}$** ：
  +  **添加$\bm{E} = \bm{A}\bm{A}^{-1}$**构造出**公因子**等适用于转化式子的项
    + 例：$\bm{B} = \bm{A}\bm{A}^{-1}\bm{B}$
+ 已知**某个矩阵的幂是$\bm{O}$ / 若干个矩阵的和或积是$\bm{O}$** ：
  + **添加$\pm\bm{O}$**等，使得**式子能够因式分解**
    + 例：已知$\bm{A}^3 = \bm{O}$，则有$\bm{B}^3 = \bm{B}^3 \pm \bm{A}^3$
+ 已知$\bm{A}^2 = \bm{A}$：
  + **构造**$(\bm{A} - a\bm{E})(\bm{A} + (a + 1)\bm{E}) = -a(a + 1)\bm{E}$

## 矩阵的秩

### 基本条件处理

+ $r(\bm{A}) = 1$：
  + 矩阵$\bm{A}$可**表示为$\bm{\alpha}\bm{\beta}^T$** ，其中$\bm{\alpha}, \bm{\beta}$为**非零列向量**（**这一约束条件不能忘！**）
+ **复杂矩阵表达式求秩**：
  1. 尝试将其**拆为若干个矩阵的积**
  2. 观察乘积表达式的项中是否有**可逆矩阵**（不妨记为$\bm{P}$、$\bm{Q}$）
  3. **利用$r(\bm{P}\bm{A}) = r(\bm{A}\bm{P}) = r(\bm{P}\bm{A}\bm{Q}) = r(\bm{A})$进行化简**


### 注意事项

+ 对一个 **$m \times n$（$m \neq n$）的矩阵**进行**初等变换**以**判断秩**：
  + 若$m > n$（**行多于列**）则使用**列初等变换**
  + 若$m < n$（**行少于列**）则使用**行初等变换**

### 部分性质的证明

+ $\bm{A}\bm{B} = \bm{O}$时，$r(\bm{A}) + r(\bm{B}) \leq n$：
  + $\bm{A}\bm{B} = \bm{O}$时，$r(\bm{A}) + r(\bm{B}) \leq r(\bm{A}\bm{B}) + n = r(\bm{O}) + n = n$
+ $\bm{A}^2 = \bm{A}$时，$r(\bm{A}) + r(\bm{A} - \bm{E}) = r(\bm{A}) + r(\bm{A} - \bm{E}) = n$：
  1. 一方面，$r(\bm{A}) + r(\bm{A} - \bm{E}) \leq r(\bm{A}(\bm{A} - \bm{E})) + n = r(\bm{A}^2 - \bm{A}) + n = n$
  2. 另一方面，$r(\bm{A}) + r(\bm{E} - \bm{A}) \geq r(\bm{A} + \bm{E} - \bm{A}) = r(\bm{E}) = n$
+ $\bm{A}^2 = \bm{E}$时，$r(\bm{A} + \bm{E}) + r(\bm{A} - \bm{E}) = r(\bm{E} + \bm{A}) + r(\bm{E} - \bm{A}) = n$：
  1. 一方面，$r(\bm{A} + \bm{E}) + r(\bm{A} - \bm{E}) \leq r((\bm{A} + \bm{E})(\bm{A} - \bm{E})) + n = r(\bm{A}^2- \bm{E}^2) + n = n$
  2. 另一方面，$r(\bm{E} + \bm{A}) + r(\bm{E} - \bm{A}) \geq r(\bm{E} + \bm{A} + \bm{E} - \bm{A}) = r(2\bm{E}) = n$

### 矩阵的组合与分块矩阵的等价变形

+ $r(\begin{bmatrix}\bm{A} & \bm{A}\bm{B}\end{bmatrix}) = r(\bm{A})$
  + 用秩的不等式 / 线性相关理论均可以证明
+ 对矩阵$\bm{A}$：
  + 通过**行变换**能得到的是$\bm{P}\bm{A}$，其中$\bm{P}$可逆
  + 通过**列变换**能得到的是$\bm{A}\bm{P}$，其中$\bm{P}$可逆
    + 推论：对分块矩阵$\begin{bmatrix} \bm{A} & \bm{O} \\ \bm{O} & \bm{B} \end{bmatrix}$：
      + 通过**行变换**能得到的是$\begin{bmatrix} \bm{A} & \bm{O} \\ \bm{P}\bm{A} & \bm{B} \end{bmatrix}$，其中$\bm{P}$可逆
      + 通过**列变换**能得到的是$\begin{bmatrix} \bm{A} & \bm{A}\bm{P} \\ \bm{O} & \bm{B} \end{bmatrix}$，其中$\bm{P}$可逆
+ **对分块矩阵进行等价变换的简单化操作方法**：
  + 将原矩阵中所有分块换成小写字母（如$\bm{A} \to a$），$\bm{E}$换成$1$，$\bm{O}$换成$0$，即**将其视作普通的若干个值**，在此基础上进行等价行变换 / 等价列变换；同时记住**没有交换律，乘在左侧的不能换到右侧，反之亦然**；变形完成后换回原来的表达


## 线性方程组与向量组

### 基本条件处理

+ **同解方程组的比较**：设$\bm{A}\bm{x} = \bm{b}_1$与$\bm{B}\bm{x} = \bm{b}_2$同解
  + 如果$\bm{A}$和$\bm{B}$的**长 $\geqslant$ 宽**，则**采用$r(\bm{A}) = r(\bm{B}) = r\left(\begin{bmatrix} \bm{A} \\ \bm{B} \end{bmatrix}\right)$**
  + 如果$\bm{A}$和$\bm{B}$的**长 $\leqslant$ 宽**，则**采用$r(\bm{A}) = r(\bm{B}) = r\left(\begin{bmatrix} \bm{A} & \bm{B} \end{bmatrix}\right)$**
+ $\bm{A} \sim \bm{B}$，则$r(\bm{A} + k\bm{E}) = r(\bm{B} + k\bm{E})$

### 注意事项

+ 对于**具体给出数值的向量组**，一定要先求出**其是否存在线性相关**！
+ 题目中指明**若干个向量线性无关**时，要注意是**单纯的“线性无关”** 还是 **“极大线性无关组”**（后者间接指明了向量构成矩阵的秩的值）
+ **基础解系**是**若干个向量**，而**不是**~~若干个向量的线性组合~~（这是 **“通解”**）！
+ **$\bm{A}^T\bm{A}\bm{x} = \bm{A}^T\bm{b}$无论什么情况（包括$A$不为方阵）都必有解**！
  + 证明：
    1. 对$\bm{A}$进行奇异值分解$\bm{A} = \bm{U}\bm{\Sigma}\bm{V}^T$，这里$\bm{U}, \bm{V}$为正交矩阵，$\bm{\Sigma} = \begin{bmatrix} \hat{\bm{\Sigma}} & \bm{O} \\ \bm{O} & \bm{O}\end{bmatrix}$，其中$\hat{\bm{\Sigma}}$为对角矩阵
    2. $\bm{A}^T\bm{A} = \bm{V}\bm{\Sigma}\bm{U}^T\bm{U}\bm{\Sigma}\bm{V}^T = \bm{V}\bm{\Sigma}^2\bm{V}^T$，原式变为$\bm{V}\bm{\Sigma}^2\bm{V}^T\bm{x} = \bm{V}\bm{\Sigma}\bm{U}^T\bm{b}$
    3. 将$\bm{\Sigma}$展开为分块矩阵形式，即$\bm{V}\begin{bmatrix} \hat{\bm{\Sigma}}^2 & \bm{O} \\ \bm{O} & \bm{O}\end{bmatrix}\bm{V}^T\bm{x} = \bm{V} \begin{bmatrix} \hat{\bm{\Sigma}} & \bm{O} \\ \bm{O} & \bm{O}\end{bmatrix} \bm{U}^T \bm{b}$
    4. 在等式两端的左侧同乘$\bm{V}^T$消去左边的$\bm{V}$，原式变为$\begin{bmatrix} \hat{\bm{\Sigma}}^2 & \bm{O} \\ \bm{O} & \bm{O}\end{bmatrix}\bm{V}^T\bm{x} = \begin{bmatrix} \hat{\bm{\Sigma}} & \bm{O} \\ \bm{O} & \bm{O}\end{bmatrix} \bm{U}^T \bm{b}$
    5. 令$\bm{V}^T\bm{x} = \begin{bmatrix} \bm{x}'_1 \\ \bm{x}'_2 \end{bmatrix}, \bm{U}^T\bm{b} = \begin{bmatrix} \bm{b}'_1 \\ \bm{b}'_2 \end{bmatrix}$，其中$\bm{x}_1$、$\bm{b}'_1$长度和$\hat{\bm{\Sigma}}$的宽度一致；原式变为$\begin{bmatrix} \hat{\bm{\Sigma}}^2 & \bm{O} \\ \bm{O} & \bm{O}\end{bmatrix}\begin{bmatrix} \bm{x}'_1 \\ \bm{x}'_2 \end{bmatrix} = \begin{bmatrix} \hat{\bm{\Sigma}} & \bm{O} \\ \bm{O} & \bm{O}\end{bmatrix} \begin{bmatrix} \bm{b}'_1 \\ \bm{b}'_2 \end{bmatrix}$
    6. 由于$\hat{\bm{\Sigma}}$、$\hat{\bm{\Sigma}}^2$均为对角阵，该方程显然能得到$\begin{bmatrix} \bm{x}'_1 \\ \bm{x}'_2 \end{bmatrix}$的解；而$\bm{V}$为正交矩阵则必可逆，从而$\bm{x} = \begin{bmatrix} \bm{x}_1 \\ \bm{x}_2 \end{bmatrix} = \begin{bmatrix} \bm{V}^{-1}\bm{x}'_1 \\ \bm{V}^{-1}\bm{x}'_2 \end{bmatrix}$必然有解


### 判断线性相关 / 无关

+ 证明**线性无关**的方法：
  + **定义法**：定义式$k_1\bm{\alpha}_1 + \cdots + k_n\bm{\alpha}_n = \bm{0}$，并证明$k_1 = \cdots = k_n = 0$
  + **矩阵法**：构造$\bm{A} = \begin{bmatrix} \bm{\alpha}_1 & \cdots & \bm{\alpha}_n \end{bmatrix}$，证明$r(\bm{A}) = n$
+ **向量长度$\leqslant 3$时**判断线性无关：
  + 直接上**矩阵法**，用**行列式$\neq 0$** 判断！
+ 判断**向量$\bm{\beta}$能否被$\bm{\alpha}_1, \bm{\alpha}_2, \cdots, \bm{\alpha}_n$表示**：
  + **观察$r(\begin{bmatrix} \bm{\alpha}_1 & \bm{\alpha}_2 & \cdots & \bm{\alpha}_n \end{bmatrix}) = r(\begin{bmatrix} \bm{\alpha}_1 & \bm{\alpha}_2 & \cdots & \bm{\alpha}_n & \bm{\beta} \end{bmatrix})$是否成立**，成立则可以被表示，不成立（前者$<$后者）则无法被表示

### 矩阵方程$\bm{A}\bm{X} = \bm{B}$

+ **$\bm{A}\bm{X} =\bm{B}$有解，等价于$r(\bm{A}) = r([\bm{A} | \bm{B}])$**
+ 求解**矩阵方程$\bm{A}\bm{X} = \bm{B}$（其中$r(\bm{A}) = n$为可逆矩阵）**：
  1. 构造$[\bm{A} | \bm{B}]$
  2. 通过**初等行变换**使该矩阵变形为$[\bm{E} | \bm{B}']$
  3. 则$\bm{X} = \bm{B}'$
+ **若$r(\bm{A}) = r < n$** ，则$\bm{A}$无法通过初等行变换变为$\bm{E}$，此时矩阵方程组解法：
  1. 构造$[\bm{A} | \bm{B}]$
  2. 通过**初等行变换（不含交换行）** 使该矩阵变形为$[\bm{A}' | \bm{B}']$，其中$\bm{A}'$为 **“行最简矩阵”**（非严格定义）：**零行在最下方，每个非零行第一个非零元素为$1$，且该非零元素所在列只能有一个非零元$1$；同时，矩阵整体排布应当遵循左上至右下的形式**
     + 以$\bm{A}' = \begin{bmatrix} 1 & 0 & 2 & 0 \\ 0 & 1 & 3 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$为例，该矩阵就是一个“行最简矩阵”
  3. 称**非零元素只有一项$1$**的列为**“基准列”**，**非零元素不小于两项**的列为 **“表示列”**
     + 在例子的$\bm{A}'$中，第1、2、4列为“基准列”，第3列为“表示列”
  4. 对特解$\bm{X'}$的每一列$\bm{x}'_i$，**取表示列的列数对应的行为$0$，剩余的行为未知数，代入$\bm{A}'\bm{x}'_i = \bm{\beta}'_i$求解$\bm{x}'_i$** ；其中$\bm{\beta}'_i$为$\bm{B}'$的第$i$列
     + 在例子的$\bm{A}'$中，$\bm{x}'_i$的待求解表达式形如$\begin{bmatrix} a \\ b \\ 0 \\ c \end{bmatrix}$（第1、2、4行为未知数，第3行为$0$），而$\bm{A}'\bm{x}'_i = \begin{bmatrix} a \\ b \\ c \\ 0 \end{bmatrix}$，结合对应列$\bm{\beta}'_i$的取值，容易求得未知数的值
  5. 考察$\bm{A}'$的各列$\bm{\alpha}'_1, \cdots, \bm{\alpha}'_n$；此时 **“表示列”的各个向量都可以用“基准列”的向量进行线性表达**，其形式为$k_1 \bm{\alpha}'_1 + k_2\bm{\alpha}'_2 + \cdots + k_r\bm{\alpha}'_r - \bm{\alpha}'_m = 0$，其中$m = r + 1, r + 2, \cdots, n$
     + 在例子的$\bm{A}'$中，有$2\bm{\alpha}'_1 + 3\bm{\alpha}'_2 - \bm{\alpha}'_3 = 0$
  6. 对每一个式子，**记$\bm{k}_m = (k_1, k_2, \cdots, k_r, 0, \cdots, 0, \underbrace{-1}_{\text{在从左至右第}m\text{位}}, 0, \cdots, 0)^T, m = r + 1, r + 2, \cdots, n$** ；此时有$\bm{A}'\bm{k}_m = \bm{0}$
     + 在例子的$\bm{A}'$中，有$\bm{k}_3 = (2, 3, -1, 0)^T$
     + 简单计算方法：**将$\bm{A}'$内部“基准列”的对角线上的$1$全部替换为$0$，“表示列”中对角线上的$0$全部替换为$-1$，然后取每个非零的第$i$列（即“表示列”）作为$\bm{k}_i$**
       + 例子中的$\bm{A}'$变形前后形如$\begin{bmatrix} 1 & 0 & 2 & 0 \\ 0 & 1 & 3 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix} \rightarrow \begin{bmatrix} 0 & 0 & 2 & 0 \\ 0 & 0 & 3 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$，取出非零的第$3$列为$\bm{k}_3 = \begin{bmatrix} 2 \\ 3 \\ -1 \\ 0 \end{bmatrix}$
  7. 最终的**通解$\bm{X}$** ：设特解$\bm{X}'$的各列为$\bm{x}'_1, \cdots, \bm{x}'_n$，则$\bm{X}$的第$i$列表达式为$$\bm{x}_i = \bm{x}'_i + \lambda_{(i, 1)}k_{r + 1} + \lambda_{(i, 2)}k_{r + 2} + \cdots + \lambda_{(i, n - r)}k_{n}$$其中$\lambda_{(i, j)}$为**任意常数**
     + 在例子的$\bm{A}'$中，有$\bm{x}_i = \bm{x}'_i + \lambda_{i1}\bm{k}_3$，其中$\lambda_{i1}$为**任意常数**

### 线性方程组的解系

+ 含 **$m$个方程、$n$个未知数（$m < n$）** 的方程组对应**矩阵为$\bm{A}$** （即$\bm{A}\bm{x} = \bm{0}$）时
  + **自由变量个数**为$n - r(\bm{A})$
  + 若**选取$x_{i_1}, x_{i_2}, \cdots, x_{i_{n - r(\bm{A})}}$为自由变量**，则$A$经过初等变换得到的**最简形式$A'$**的**第$i_1, i_2, \cdots, i_{n - r(\bm{A})}$列应当线性无关**（即行列式$ \neq 0$）
+ 设$\bm{A}$为$m \times n$矩阵，
  + 若$\bm{A}\bm{x} = \bm{0}$的**基础解系包含$a$个向量**，则$r(\bm{A}) = n - a$
  + **基础解系大小$a = n - r(\bm{A})$**
    + 注意：以上计算都**只考虑$\bm{A}$的列数$n$** ，和$\bm{A}$的行数$m$、$m$与$n$的大小关系均**无关**！
+ 选择题中**已知若干个解$\bm{x}$** ，**判断哪个系数矩阵$\bm{A}$满足$\bm{A}\bm{x} = \bm{0}$** ：
  1. 找出**解中的极大线性无关组**，或**通过组合得到极大线性无关组**
  2. 根据**极大线性无关组的大小$a$** ，判断 **$\bm{A}$是否满足$r(\bm{A}) \leq n - a$** ；
     + 若**指明原来的解包含（整个）基础解系**，则$r(\bm{A}) = n - a$
  3. 在此基础上，**代入解**判断**是否每个解都满足$\bm{A}\bm{x} = \bm{0}$**

### 线性方程组的几何意义

+ 对线性方程组$$\begin{cases} a_1 x + b_1 y + c_1 z = d_1 \\ a_2 x + b_2 y + c_2 z = d_2 \\ a_3 x + b_3 y + c_3 z = d_3 \end{cases}$$令$\bm{A} = \begin{bmatrix} a_1 & b_1 & c_1 \\ a_2 & b_2 & c_2 \\ a_3 & b_3 & c_3 \end{bmatrix}$，$\bar{\bm{A}} = \begin{bmatrix} a_1 & b_1 & c_1  & d_1 \\ a_2 & b_2 & c_2 & d_2 \\ a_3 & b_3 & c_3 & d_3 \end{bmatrix}$，则有

| $r(\bm{A})$ | $r(\bar{\bm{A}})$ |                     额外信息                     |               图形               |
| :---------: | :---------------: | :----------------------------------------------: | :------------------------------: |
|      3      |         3         |                                                  |        三张平面相交于一点        |
|      2      |         2         |              任意两个向量都线性无关              |      三张平面相交于一条直线      |
|      2      |         2         | 有两个向量线性相关（一个向量是另一个向量的倍数） |   一张平面与两张重合的平面相交   |
|      1      |         1         |                                                  |           三张平面重合           |
|      2      |         3         |              任意两个向量都线性无关              | 三张平面两两相交，且交线互相平行 |
|      2      |         3         | 有两个向量线性相关（一个向量是另一个向量的倍数） |   一张平面与两张平行的平面相交   |
|      1      |         2         |              任意两个向量都线性无关              |       三张平面平行但不重合       |
|      1      |         2         | 有两个向量线性相关（一个向量是另一个向量的倍数） | 两张平面重合，第三张平面与之平行 |

## 特征值

### 基本条件处理

+ **求解伴随矩阵$\bm{A}^\text{*}$的特征值**：
  + 通过$\bm{A}^\text{*} = |\bm{A}|\bm{A}^{-1}$从而有$\lambda^\text{*}_i = \frac{|\bm{A}|}{\lambda_i} = \frac{\prod\limits_{j = 1}^n \lambda_j}{\lambda_i} = \prod\limits_{j = 1 \atop j \neq i}^n \lambda_j$求解
+ **已知伴随矩阵$\bm{A}^\text{*}$的特征值$\lambda^\text{*}_0$** ：
  + $\bm{A}$的特征值**包含$\lambda = \frac{|\bm{A}|}{\lambda^\text{*}_0} = \frac{|\bm{A}^\text{*}|^{\frac{1}{n - 1}}}{\lambda^\text{*}_0}$**
+ 特征值$\lambda$的**几何重数**的求解：
  + **计算$n - r(\lambda\bm{E} - \bm{A})$**

### 注意事项

+ 若$\bm{A}$可经由可逆矩阵$\bm{P}$对角化为$\bm{\Lambda} = \bm{P}^{-1}\bm{A}\bm{P}$（即$\bm{A} = \bm{P}\bm{\Lambda}\bm{P}^{-1}$），则$\bm{\Lambda}$的每个**对角线元素**均为**特征值**，$\bm{P}$的每个**列向量**均为**特征向量**，且二者**从左到右一一对应**
+ 对抽象矩阵，**特征值取值范围满足$\lambda \in \{a_1, a_2, \cdots, a_n\}$不代表取值范围内每个值都有对应特征值！**
  + 如：$\lambda^2 = 1$**不代表**~~必定有$\lambda = 1$和$\lambda = -1$~~
+ 对$n \times n$的矩阵$\bm{A}$，$r(\bm{A}) = r < n$可推出$0$**至少**为$n - r$重特征值！
  + 注意**不是**~~$0$为$n - r$重特征值~~
+ 上一条推论：**秩**和**非零特征值个数**是**不存在一一对应关系的**！
  + **正负惯性指数**才是和**非零特征值个数**直接相关的！


### 行列式与特征值

+ 对**抽象矩阵**（不知道具体形式的矩阵），证明$|\lambda\bm{E} - \bm{A}| = 0$：
  + 考虑**证明$|\lambda\bm{E} - \bm{A}| = -|\lambda\bm{E} - \bm{A}|$**
    + 可以推广至**任何需要证明（抽象矩阵的）$|\bm{A}| = 0$** 的情形
+ **出现$|a\bm{A} + b\bm{E}| = 0$** ：
  + $-\frac{a}{b}$是$\bm{A}$的特征值

### 矩阵多项式与特征值

+ 已知$\bm{A}^n = \bm{O}, \bm{A}^n = \pm \bm{E}$：
  + **直接求出$\bm{A}$的特征值**：
    + $\bm{A}^n = \bm{O} \Rightarrow \lambda = 0$
    + $\bm{A}^n = \pm \bm{E} \Rightarrow \lambda^n = \pm 1$
+ **求解$|f(\bm{A})|$，其中$f$为多项式函数**（如$\bm{A}^2 + \bm{A} + \bm{E}$）：
  1. **求出 / 利用$\bm{A}$的所有特征值**
  2. 对多项式$f$和$\bm{A}$的某个特征值$\lambda$，$f(\bm{A})$的特征值为$f(\lambda)$
  3. 最后$|f(\bm{A})| = \prod\limits_i f(\lambda_i)$
+ 见到 **$f(\bm{A}) = \bm{O}$，其中$f$为多项式函数**：
  1. 对多项式$f$和$\bm{A}$的某个特征值$\lambda$，$f(\bm{A})$的特征值为$f(\lambda)$
  2. 用$f(\lambda) = 0$求出所有**可能**的$\lambda$
     + 注意是**可能**的$\lambda$而**不是**~~实际存在的$\lambda$~~！特征值取值范围满足$\lambda \in \{a_1, a_2, \cdots, a_n\}$不代表取值范围内每个值都有对应特征值（参见本节注意事项）
       + 例：$\lambda^2 = 1$**不代表**~~必定有$\lambda = 1$和$\lambda = -1$~~
  3. 根据题目条件进行调整；如$\lambda = \pm 1$，则考虑构造出$\lambda^2 \equiv 1$

## 特征向量

### 基本条件处理

+ **$\bm{\xi}$是$\bm{A}$的属于$\lambda_0$的特征向量**，等价于 **$\bm{\xi}$是$(\lambda_0\bm{E} - \bm{A})\bm{x} = \bm{0}$的非零解**
+ 若 **$\bm{\alpha}$为$\bm{A}$的特征向量**（对应**特征值为$\lambda$** ），则 **$\bm{\alpha}$也为$\bm{A} + k\bm{E}$的特征向量**（对应**特征值为$\lambda + k$** ）

### 注意事项

+ **特征向量的$k$倍不一定仍然是特征向量**；**必须要求$k \neq 0$** ！
  + 如果是**同一个特征值下多个特征向量**构成的**线性空间**，则其**系数必须不全为零**！
+ 对于**实对称矩阵**，**不同特征值对应的特征向量**必定**相互正交**！
  + 如果正交矩阵$\bm{Q}$和实对称矩阵$\bm{A}$满足$\bm{Q}^{-1}\bm{A}\bm{Q} = \bm{\Lambda}$（$\bm{\Lambda}$为 **（特征值构成的）对角矩阵**），则 **$\bm{Q}$各列**为 **$\bm{\Lambda}$相应列的特征值对应的特征向量“单位化”后**的结果！
    + 由于实对称矩阵的特征向量互相正交，故只需要**单位化特征向量**，就能完成所有特征向量的**规范正交化**
+ **特征向量乘以$k$（$k \neq 0$）倍，对应特征值不变**！
+ **同一特征值的不同特征向量相加 / 相减 / 进行线性组合，对应特征值不变**！
+ 矩阵**代入多项式函数$f(\bm{A})$**、**取逆$\bm{A}^{-1}$**、以及**这二者的任意组合（如$(\bm{A} + \bm{E})^{-1}$）**均**不改变特征向量**！

### 求解特征向量

+ 已知$\bm{A}$的特征值$\lambda$，求解特征向量：
  1. 计算$\lambda\bm{E} - \bm{A}$，记为$\bm{B}$
  2. 通过**初等行变换（不含交换行）** 使该矩阵变形为$\bm{B}'$，其中$\bm{B}'$为 **“行最简矩阵”**（非严格定义）：**零行在最下方，每个非零行第一个非零元素为$1$，且该非零元素所在列只能有一个非零元$1$；同时，矩阵整体排布应当遵循左上至右下的形式**
     + 以$\bm{B}' = \begin{bmatrix} 1 & 0 & 2 & 0 \\ 0 & 1 & 3 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$为例，该矩阵就是一个“行最简矩阵”
  3. 称**非零元素只有一项$1$**的列为**“基准列”**，**非零元素不小于两项**的列为 **“表示列”**
     + 在例子的$\bm{B}'$中，第1、2、4列为“基准列”，第3列为“表示列”
  4. 考察$\bm{B}'$的各列$\bm{\alpha}'_1, \cdots, \bm{\alpha}'_n$；此时 **“表示列”的各个向量都可以用“基准列”的向量进行线性表达**，其形式为$k_1 \bm{\alpha}'_1 + k_2\bm{\alpha}'_2 + \cdots + k_r\bm{\alpha}'_r - \bm{\alpha}'_m = 0$，其中$m = r + 1, r + 2, \cdots, n$
     + 在例子的$\bm{B}'$中，有$2\bm{\alpha}'_1 + 3\bm{\alpha}'_2 - \bm{\alpha}'_3 = 0$
  5. 对每一个式子，**记$\bm{k}_m = (k_1, k_2, \cdots, k_r, 0, \cdots, 0, \underbrace{-1}_{\text{在从左至右第}m\text{位}}, 0, \cdots, 0)^T, m = r + 1, r + 2, \cdots, n$** ；此时有$\bm{B}'\bm{k}_m = \bm{0}$
     + 在例子的$\bm{B}'$中，有$\bm{k}_3 = (2, 3, -1, 0)^T$
     + 简单计算方法：**将$\bm{B}'$内部“基准列”的对角线上的$1$全部替换为$0$，“表示列”中对角线上的$0$全部替换为$-1$，然后取每个非零的第$i$列（即“表示列”）作为$\bm{k}_i$**
       + 例子中的$\bm{B}'$变形前后形如$\begin{bmatrix} 1 & 0 & 2 & 0 \\ 0 & 1 & 3 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix} \rightarrow \begin{bmatrix} 0 & 0 & 2 & 0 \\ 0 & 0 & 3 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$，取出非零的第$3$列为$\bm{k}_3 = \begin{bmatrix} 2 \\ 3 \\ -1 \\ 0 \end{bmatrix}$
  6. 此时**每个$\bm{k}_i$即为特征向量**！
     + 如果是**实对称矩阵**，记得进行**单位化**以方便得到正交矩阵！

### 从已知特征向量推导未知特征向量

+ 对一般的矩阵，使用**不同特征值对应的特征向量线性无关**
+ 对**实对称矩阵**，使用**不同特征值对应的特征向量正交**
  + 在此基础上，可以对特征向量进行**单位化**，如此得到**正交矩阵**$\bm{P}$后能够直接求解$\bm{A} = \bm{P}\bm{\Lambda}\bm{P}^{-1} = \bm{P}\bm{\Lambda}\bm{P}^T$

## 相似理论

### 基本条件处理

+ 若已知部分项含未知数的两个矩阵$\bm{A}, \bm{B}$**相似**，快速求解参数：
  + 使用$|\bm{A}| = |\bm{B}|$、$\text{tr}(\bm{A}) = \text{tr}(\bm{B})$

### 注意事项

+ 对多项式函数$f$，若$f(\bm{A}) = \bm{O}$，则$\bm{A}$的**所有特征值**必定**满足$f(\lambda) = 0$** ；同理，若$f(\bm{A}) = \bm{E}$，则$\bm{A}$的**所有特征值**必定**满足$f(\lambda) = 1$**
+ 若**两个矩阵$\bm{A}, \bm{B}$均可相似对角化**，则**“可以用同一个可逆矩阵$P$相似对角化 $\Leftrightarrow$ 两个矩阵特征向量相同”**！
  + 利用接下来“特征值相同 / 特征向量相同的判断 / 证明”一节中“判定$\bm{A}\bm{B} = \bm{B}\bm{A}$时，特征向量相同”的证明方法可类似证明

### 判断相似

+ **快速判断**$\bm{A}$是否**和一个已知的对角阵$\bm{\Lambda}$相似**：依次尝试方法
  1. 比较矩阵的**迹**是否**相同**
  2. 对**三阶及以下**的矩阵：
     + 计算**行列式**是否**相同**
  3. 对角阵$\bm{\Lambda}$存在**重复特征值**
     + 对于对角阵$\bm{\Lambda}$中**重复$n_i$次的特征值$\lambda_i$** ，计算 **$n - r(\lambda_i \bm{E} - \bm{A})$是否等于$n_i$**
  4. 以上快速判断法失效后，再利用$|\lambda_i \bm{E} - \bm{A}|$对**每个特征值进行一一对照检验**
+ **判断$\bm{A}$**是否**和$\bm{B}$相似**：
  + 存在**可逆矩阵$\bm{C}$使得$\bm{C}^{-1}\bm{A}\bm{C} = \bm{B}$**
  + **传递性**：$\bm{A}$合同于$\bm{B}$，$\bm{B}$合同于$\bm{C}$，则$\bm{A}$合同于$\bm{C}$
  + 检查$\bm{A}$和$\bm{B}$，其**所有特征值**和**每个特征值的几何重数相同**
    + 判断特征值$\lambda$的**几何重数**是否相同：**计算$r(\lambda \bm{E} - \bm{A})$、$r(\lambda \bm{E} - \bm{B})$是否相同**；实际几何重数为$n - r(\lambda \bm{E} - \bm{A})$、$n - r(\lambda \bm{E} - \bm{B})$
  
+ 快速**判断$\bm{A}$**是否**不和$\bm{B}$相似**：
  + $r(\bm{A}) \neq r(\bm{B})$
  + $|\bm{A}| \neq |\bm{B}|$
  + $\text{tr}(\bm{A}) \neq \text{tr}(\bm{B})$
  + $\bm{A}$与$\bm{B}$至少一个特征值不同


### 判断 / 证明 / 已知相似对角化

+ 快速**判断$\bm{A}$**是否可以**相似对角化**：
  1. **实对称矩阵必定能相似对角化**
  2. **对上 / 下三角阵，其特征值即为对角线元素，若对角线元素（特征值）各不相同则可以相似对角化**
  3. 以上快速判断法失效后，再**利用“每个特征值$\lambda_i$的几何重数$n - r(\lambda_i \bm{E} - \bm{A})$应当和代数重数相等”对每个特征值进行一一检验**
+ 证明**不可相似对角化**通常用**反证法**
  + **对角元素相同、非对角阵的上 / 下三角阵**必定不可相似对角化
+ **已知可以相似对角化**，求解参数：
  + 利用**特征值的代数重数等于几何重数$n - r(\lambda \bm{E} - \bm{A})$**


### 特征值相同 / 特征向量相同的判断 / 证明

+ 证明 **$\bm{A}, \bm{B}$特征值相同**（此处两个矩阵可以为表达式，如$f(\bm{A}, \bm{B}), g(\bm{A}, \bm{B})$）：
  + 在$\bm{A}\bm{x} = \lambda \bm{x}$的基础上**想办法得到$\bm{B}\bm{C}\bm{x} = \lambda \bm{C}\bm{x}$**
+ 判定 **$\bm{A}\bm{B} = \bm{B}\bm{A}$时，特征向量相同**：
  + $\bm{A}$、$\bm{B}$**均可相似对角化** $\Rightarrow$ $\bm{A}$、$\bm{B}$特征向量相同
  + $\bm{A}$**特征值互异** $\Rightarrow$ $\bm{A}$、$\bm{B}$特征向量相同
    + 证明：对$\bm{A}$关于特征值$\lambda$的特征向量$\bm{\alpha}$，由$\bm{A}\bm{B}\bm{\alpha} = \bm{B}\bm{A}\bm{\alpha} = \lambda\bm{B}\bm{\alpha}$后分别讨论：
      + $\bm{B}\bm{\alpha} = \bm{0}$，则$0$为$\bm{B}$特征值，$\bm{\alpha}$也为$\bm{B}$特征向量
      + $\bm{B}\bm{\alpha} \neq \bm{0}$，$\bm{B}\bm{\alpha}$为$\bm{A}$关于特征值$\lambda$的特征向量，而$\bm{A}$特征值互异（即每个特征值代数重数为$1$，可知几何重数也为$1$），从而$\bm{B}\bm{\alpha}$与$\bm{\alpha}$线性相关（即$\bm{B}\bm{\alpha} = \mu\bm{\alpha}$），最终得到$\bm{\alpha}$也为$\bm{B}$特征值

## 二次型

### 基本条件处理

+ **二次型矩阵为$\bm{A}^T\bm{A}$、$\bm{\alpha}\bm{\alpha}^T$等类似形式**：
  + **代入原二次型**，有
    + $\bm{x}^T\bm{A}^T\bm{A}\bm{x} = (\bm{A}\bm{x})^T\bm{A}\bm{x} = \parallel \bm{A}\bm{x} \parallel^2$
    + $\bm{x}^T\bm{\alpha}\bm{\alpha}^T\bm{x} = (\bm{\alpha}^T\bm{x})^T\bm{\alpha}^T\bm{x} = (\bm{\alpha}^T\bm{x})^2$

### 注意事项

+ **标准型 / 规范型**是**表达式**；标准型 / 规范型**对应的矩阵**才是**矩阵**！ 
+ 如果**已知二次型矩阵$\bm{A}$与对角阵$\bm{\Lambda}$相似**（即$\bm{Q}^T\bm{A}\bm{Q} = \bm{\Lambda}$，亦即**已知所有特征值**），则 **$\bm{x}^T\bm{A}\bm{x}$等价于$\bm{y}^T\bm{\Lambda}\bm{y}$** ，其**取值范围等完全相等**
  + 此时$\bm{y} = \bm{Q}\bm{x}$


### 正交变换与正交矩阵的求解

+ 已知实对称矩阵$\bm{A}$，**求解正交矩阵$\bm{Q}$** 使得其可以相似对角化为$\bm{Q}^T\bm{A}\bm{Q} = \bm{\Lambda}$，只能使用**正交变换法**：
  1. 计算$\bm{A}$的**所有特征值$\lambda_i$**
  2. **对每个$\lambda_i$计算对应的特征向量$\bm{\xi}_j$** （列向量），并将其**单位化**为$\bm{\alpha}_j = \frac{\bm{\xi}_j}{|\bm{\xi}_j|}$
     + 由于实对称矩阵的特征向量互相正交，故只需要单位化特征向量，就能完成所有特征向量的规范正交化
  3. $[\bm{\alpha}_1, \bm{\alpha}_2, \cdots, \bm{\alpha}_n]$构成正交矩阵$\bm{Q}$

### 二次型的求解

+ 对**表达式**：
  + **配方法（换元变形法）**：
    1. **换元出$y_i = a_{i1}x_1 + \cdots + a_{in}x_n, i = 1, 2, \cdots, n$** ，其中$y_i$使原二次型变为$\lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2$的形式
    2. 解联立方程，**化为$x_i = b_{i1}y_1 + \cdots + b_{in}y_n, i = 1, 2, \cdots, n$的形式**
    3. 将**这里的系数$b_{ij}$**作为**矩阵$\bm{x} = \bm{C}\bm{y}$第$i$行第$j$列的元素**

  + **常见配方法形式**：
    + $x_1x_2$ $\rightarrow$ $\begin{cases}x_1 = y_1 + y_2 \\ x_2 = y_1 - y_2\end{cases}$ $\rightarrow$ $\begin{cases}y_1 = \frac{1}{2}(x_1 + x_2) \\ y_2 = \frac{1}{2}(x_1 - x_2)\end{cases},\quad x_1x_2 = y_1^2 - y_2^2$
    + $(ax_1 + bx_2)(cx_1 + dx_2)$ $\rightarrow$ $\begin{cases}y_1 = ax_1 + bx_2 \\ y_2 = cx_1 + dx_2\end{cases},\quad (ax_1 + bx_2)(cx_1 + dx_2)= y_1y_2$  $\rightarrow$ $\begin{cases}z_1 = \frac{1}{2}(y_1 + y_2) \\ z_2 = \frac{1}{2}(y_1 - y_2)\end{cases},\quad y_1y_2 = z_1^2 - z_2^2$，同时$\begin{bmatrix} z_1 \\ z_2 \end{bmatrix} = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} \end{bmatrix} \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} \frac{1}{2}(a + c) & \frac{1}{2}(b + d) \\ \frac{1}{2}(a - c) & \frac{1}{2}(b - d) \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$

+ 对**矩阵**：使用**正交变换法**
  1. 计算二次型矩阵$\bm{A}$的**由单位化特征向量构成的正交阵$\bm{Q}$** 和对应的、**由特征值$\lambda_1, \Lambda_2, \cdots, \lambda_n$构成的对角阵$\bm{\Lambda}$**
  2. 根据题目要求获得最终变换：
       + 最终目标为得到**标准型**，则
         + 变换$\bm{x} = \bm{C}\bm{y}$的$\bm{C} = \bm{Q}$
         + 标准型为$f = \lambda_1y_1^2 + \cdots + \lambda_ny_n^2$
       + 最终目标为得到**规范型**，则
         + 变换$\bm{x} = \bm{C}\bm{y}$的$\bm{C} = \bm{Q}\bm{\Lambda}^{-1}$
         + 规范型为$f = \text{sgn}(\lambda_1)y_1^2 + \cdots + \text{sgn}(\lambda_n)y_n^2$


### 合同的判断与求解

+ **判断$\bm{A}$和$\bm{B}$合同**：
  + 存在可逆矩阵$\bm{C}$使得$\bm{C}^T\bm{A}\bm{C} = \bm{B}$
  + **传递性**：$\bm{A}$合同于$\bm{B}$，$\bm{B}$合同于$\bm{C}$，则$\bm{A}$合同于$\bm{C}$
  + **正、负惯性指数相同**（求解$|\lambda\bm{E} - \bm{A}| = 0$观察解的正负性）
  + **同阶实对称矩阵满足相似**
    + 注意**合同不一定相似**，**只有同阶实对称矩阵合同**才能确保相似
+ **已知$\bm{A}$、$\bm{B}$合同**，求解$\bm{C}^T\bm{A}\bm{C} = \bm{B}$的$\bm{C}$：
  + **配方法**：先配出一个参数（如$x_1$），再配出下一个参数（如$x_1 + x_2$），依此类推
  + **成对初等变换**：
    + **直接变换法**：
      1. 构造$[\bm{A} | \bm{E}]$
      2. 每次变换**对行操作完后，对称地对左半部分的列操作一次**
         + **对列操作不影响右半部分**；每次操作结束后，左半部分应当是一个实对称矩阵
      3. 最后**得到$[\bm{B} | \bm{C}^T]$**
         + 注意**不是**$[\bm{B} | \bm{C}]$！
    + **间接变换法**：
      1. 构造$[\bm{A} | \bm{E}]$、$[\bm{B}|\bm{E}]$
      2. **先对$[\bm{B}|\bm{E}]$操作**；每次变换**对行操作完后，对称地对左半部分的列操作一次**
      3. 目标为从$[\bm{B}|\bm{E}]$得到$[\bm{\Lambda}|\bm{C}_B^T]$，其中 **$\bm{\Lambda}$为一个对角阵**，而 **$\bm{C}_B^T$的形式应当尽量简洁**！
         + $\bm{\Lambda}$只要是对角阵就行，不需要凑成和特征值相关的对角阵！**重点是让$\bm{C}_B^T$的形式尽量简洁**！
      4. **计算$\bm{C}_B^T$的逆$(\bm{C}_B^{-1})^T$**
         + 只有将$\bm{C}_B^T$形式变得尽量简洁，这里的计算量才会降低！
      5. **再对$[\bm{A}|\bm{E}]$操作**；每次变换**对行操作完后，对称地对左半部分的列操作一次**
      6. 目标为从$[\bm{A}|\bm{E}]$得到$[\bm{\Lambda}|\bm{C}_A^T]$
         + 这里$\bm{\Lambda}$是先前求出来的已知$\bm{\Lambda}$，而不是~~任意一个对角阵~~
      7. 最后得到$\bm{C}^T = (\bm{C}_B^{-1})^T\bm{C}_A^T$，即$\bm{C} = \left((\bm{C}_B^{-1})^T\bm{C}_A^T\right)^T$

### 二次型正定的判定

+ 对**表达式**：
  1. **找特殊值**，如果**特殊值能使得其取值变为$0$或$\leqslant 0$** ，则该二次型**不正定**；其他情况亦然
     +  例：$(x_1 - x_2)^2 + (x_2 - x_3)^2 + (x_3 - x_1)^2$存在$(x_1, x_2, x_3) = (1, 1, 1)$使其取值为$0$，故不正定
  2. 转**矩阵处理方法**
+ 对**矩阵**：
  + 判定**正定**：
    + 对**一般**实对称矩阵，**计算各阶顺序主子式是否均$> 0$**
    + 若**已知所有特征值**，则**所有特征值均$> 0$**
  + 判定**不正定**：**违反以下任意一条**
    + $a_{ii} > 0$，即**对角线上元素均$> 0$**
    + $|\bm{A}| > 0$，或扩张版的**各阶顺序主子式均$> 0$**

