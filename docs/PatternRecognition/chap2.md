# 统计决策方法

## 引言
假定样本$\bm{x} \in R^d$是由$d$维实数特征组成的，即$\bm{x} = [x_1, x_2, ..., x_d]^T$；
假定要研究的类别有$c$个，记为$\omega_i,\ i=1,2,...,c$，类别数$c$和各类别的先验概率$P(\omega_i)$已知。假定各类中样本的分布密度，即类条件密度$p(\bm{x}|\omega_i)$已知。
**决策：** 对某个未知样本$\bm{x}$，判断其属于哪一类。

对二分类问题，在样本$\bm{x}$上错误的概率为
$$p(e|\bm{x}) = \begin{cases}
P(\omega_2|\bm{x}) & \bm{x} \in \omega_1 \\
P(\omega_1|\bm{x}) & \bm{x} \in \omega_2 \\
\end{cases}$$

**错误率**定义为所有服从同样分布的独立样本上错误概率的期望，即
$$P(e) = \int P(e|\bm{x})p(\bm{x})\,d\bm{x}$$

**正确率**即作出正确决策的概率，通常记作$P(c)$；显然$P(c) = 1 - P(e)$。

## 最小错误率贝叶斯决策
注：无特殊说明的贝叶斯决策即指最小错误率贝叶斯决策。

最小错误率的目标就是**求解一种决策规则，使得错误率$P(e)$最小化，即**$$\min \quad P(e) = \int P(e|\bm{x})p(\bm{x})\,d\bm{x}$$

注意到对所有$\bm{x}$，有$P(e|\bm{x}) \geqslant 0, P(x) \geqslant 0$，所以上式等价于对所有$\bm{x}$最小化$P(e|\bm{x})$。

### 二分类问题的最小错误率贝叶斯决策
回顾二分类问题的错误概率$p(e|\bm{x}) = \begin{cases}
P(\omega_2|\bm{x}) & \bm{x} \in \omega_1 \\
P(\omega_1|\bm{x}) & \bm{x} \in \omega_2 \\
\end{cases}$，可知使得错误率最小的分类决策就是使后验概率最大的决策。

因此，对**二分类问题**，决策规则为：
> 如果$P(\omega_1|\bm{x}) > P(\omega_2|\bm{x})$，则$\bm{x} \in \omega_1$；反之，$\bm{x} \in \omega_2$。

或简记为
> 如果$P(\omega_1|\bm{x}) \lessgtr P(\omega_2|\bm{x})$，则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$

其中，后验概率用贝叶斯公式即可求得：
$$P(\omega_i | \bm{x}) = \dfrac{p(\bm{x} | \omega_i)P(\omega_i)}{p(\bm{x})} = \dfrac{p(\bm{x} | \omega_i)P(\omega_i)}{\sum\limits_{j = 1}^2 p(\bm{x} | \omega_j)P(\omega_j)}$$

等价形式：
+ 如果满足$$P(\omega_i | \bm{x}) = \max\limits_{j = 1, 2} P(\omega_j | \bm{x})$$则$\bm{x}\in \omega_i$；
+ 由于上式中$P(\omega_i | \bm{x})$分母相同，可以只比较分子，即：
  如果满足$$P(\bm{x} | \omega_i)P(\omega_i) = \max\limits_{j = 1, 2}P(\bm{x} | \omega_j)P(\omega_j)$$则$\bm{x}\in \omega_i$；
+ 由于先验概率$P(\omega_i)$是固定的，与$\bm{x}$无关，因此可转化为以下形式：
  如果满足$$l(\bm{x}) = \dfrac{p(\bm{x}|\omega_1)}{p(\bm{x}|\omega_2)} \lessgtr \lambda = \dfrac{P(\omega_2)}{P(\omega_1)}$$则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$，其中$l(\bm{x}) = \dfrac{p(\bm{x}|\omega_1)}{p(\bm{x}|\omega_2)}$称为**似然比**；
+ 为方便计算，定义**对数似然比**（注意这里取的负对数）：$$h(\bm{x}) = -\ln [l(\bm{x})] = -\ln p(\bm{x}|\omega_1) + \ln p(\bm{x}|\omega_2)$$决策规则转化为以下形式：
  如果满足$$h(\bm{x}) \lessgtr \ln \dfrac{P(\omega_1)}{P(\omega_2)}$$则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$；

### 多分类问题的最小错误率贝叶斯决策
> 如果满足$$P(\omega_i | \bm{x}) = \max_{j=1, ..., c} P(\omega_j | \bm{x})$$则$\bm{x} \in \omega_i$。

等价：
> 如果满足$$p(\bm{x} | \omega_i) P(\omega_i) = \max_{j=1, ..., c} p(\bm{x} | \omega_j)P(\omega_j)$$则$\bm{x} \in \omega_i$。

在多分类问题中，需要把特征空间分成$\mathscr{R}_1, \mathscr{R}_2, ..., \mathscr{R}_c$共$c$个区域。其平均错误概率为
> $$P(e) = \sum_{i = 1}^c \sum_{\substack{j = 1 \\ j \neq i}}^c [P(\bm{x} \in \mathscr{R}_j | \omega_j)] P(\omega_i)$$

该式计算量较大，可以用**平均正确率**间接计算：
> $$P(c) = \sum_{j = 1}^c P(\bm{x} \in \mathscr{R}_j | \omega_j)P(\omega_j) = \sum_{j = 1}^c \int_{\mathscr{R}_j} p(\bm{x} | \omega_j)P(\omega_j)\,d\bm{x} \\ P(e) = 1 - P(c) = 1 - \sum_{j = 1}^c \int_{\mathscr{R}_j} p(\bm{x} | \omega_j)P(\omega_j)\,d\bm{x}$$

## 最小风险贝叶斯决策
问题的重新表述：
1. 将样本$\bm{x}$看做$d$维随机向量$\bm{x} = [x_1, x_2, ..., x_d]^T$
2. 状态空间$\varOmega$由$c$个可能的状态（$c$类）组成：$\varOmega = \{\omega_1, \omega_2, ..., \omega_c\}$
3. 对随机向量$\bm{x}$可能采取的$k$个决策组成了决策空间$\mathscr{A} = \{\alpha_1, \alpha_2, ..., \alpha_k\}$
    + 此处未假定$k = c$；原因在于除了判别为某一类，还可能存在决策为全部拒绝（无法判断某一类）、将几类合并为一个大类的决策。
4. 对实际状态为$\omega_j$的向量$\bm{x}$，采取决策$\alpha_j$所带来的损失$$\lambda(\alpha_i, \omega_j),\quad i = 1, ..., k,\quad j = 1, ..., c$$称为**损失函数**。其通常能以$k \times c$的**决策表**的形式给出。

> 对于每个样本$\bm{x}$，对其采用决策$\alpha_i,\ i=1, ..., k$的**期望损失**是$$R(\alpha_i | \bm{x}) = E[\lambda(\alpha_i, \omega_j) | \bm{x}] = \sum_{j = 1}^c \lambda(\alpha_i, \omega_j) P(\omega_j | \bm{x}),\quad i = 1, ..., k$$

> 对某一个决策规则$\alpha(\bm{x})$，其对特征空间中所有可能的样本$\bm{x}$采取决策所造成的期望损失是$$R(\alpha) = \int R(\alpha(\bm{x})| \bm{x}) p(\bm{x})\,d\bm{x}$$称为**平均风险**或**期望风险**。**最小风险贝叶斯决策即为最小化这一期望风险**，即求$$\alpha = \arg\min R(\alpha)$$

由于$R(\alpha(\bm{x})| \bm{x}), p(\bm{x})$均非负，且$p(\bm{x})$已知，故使积分和最小，就是对所有$\bm{x}$使得$R(\alpha(\bm{x})| \bm{x})$最小。

因此，**最小风险贝叶斯决策就是：**
> 如果满足$$R(\alpha_i | \bm{x}) = \min_{j = 1, ..., k} R(\alpha_j | \bm{x})$$则$\alpha = \alpha_i$。

**计算步骤:**
> 1. 计算后验概率：$$P(\omega_i | \bm{x}) = \dfrac{p(\bm{x} | \omega_i)P(\omega_i)}{\sum\limits_{j = 1}^c p(\bm{x} | \omega_j)P(\omega_j)}, \quad j = 1, ..., c$$
> 2. 利用决策表，计算条件风险$$R(\alpha_i | \bm{x}) = \sum_{j = 1}^c \lambda(\alpha_i, \omega_j) P(\omega_j | \bm{x}),\quad i = 1, ..., k$$
> 3. 在各种决策中选择风险最小的决策：$$\alpha = \arg\min_{i = 1, 2, ..., k} R(\alpha_i | \bm{x})$$

$\blacktriangle$当取决策与状态相同的损失为0，决策与状态不同的损失为1时，最小风险贝叶斯决策转化为最小错误率贝叶斯决策。

### 二分类的最小风险贝叶斯决策
如果满足$$\lambda_{11}P(\omega_1 | \bm{x}) + \lambda_{12}P(\omega_2 | \bm{x}) \lessgtr \lambda_{21}P(\omega_2 | \bm{x}) + \lambda_{22}P(\omega_2 | \bm{x})$$则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$

判别条件的等价表达：
+ $$(\lambda_{11} - \lambda_{21})P(\omega_1 | \bm{x}) \lessgtr (\lambda_{22} - \lambda_{12})P(\omega_2 | \bm{x})$$
+ $$\dfrac{P(\omega_1 | \bm{x})}{P(\omega_2 | \bm{x})} = \dfrac{p(\bm{x} | \omega_1)P(\bm{x})}{p(\bm{x} | \omega_2)P(\bm{x})} \lessgtr \dfrac{\lambda_{22} - \lambda_{12}}{\lambda_{11} - \lambda_{21}} = \dfrac{\lambda_{12} - \lambda_{22}}{\lambda_{21} - \lambda_{11}}$$
+ $$l(\bm{x}) = \dfrac{p(\bm{x}|\omega_1)}{p(\bm{x}|\omega_2)} \lessgtr \dfrac{P(\omega_2)}{P(\omega_1)}\cdot \dfrac{\lambda_{12} - \lambda_{22}}{\lambda_{21} - \lambda_{11}}$$

## 两类错误率，Neyman-Pearson决策与ROC曲线
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">决策</th>
    <th class="tg-0pky" colspan="2">状态</th>
  </tr>
  <tr>
    <th class="tg-0pky">阳性</th>
    <th class="tg-0pky">阴性</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">阳性</td>
    <td class="tg-0pky">真阳性（TP）</td>
    <td class="tg-0pky">假阳性（FP）</td>
  </tr>
  <tr>
    <td class="tg-0pky">阴性</td>
    <td class="tg-0pky">假阴性（FN）</td>
    <td class="tg-0pky">真阴性（TN）</td>
  </tr>
</tbody>
</table>
T: True, F: False, P: Positive, N: Negative

**灵敏度：**$\text{Sn} = \dfrac{\text{TP}}{\text{TP} + \text{FN}}$，**真正阳性样本中，检测为阳性所占的比例**；

**特异度：**$\text{Sp} = \dfrac{\text{TN}}{\text{TN} + \text{FP}}$，**真正阴性样本中，检测为阴性的比例**；

**第一类错误率 / 假阳性率：**$\alpha = \dfrac{\text{FP}}{\text{TN} + \text{FP}}$，满足$\text{Sp} = 1 - \alpha$，**真正阴性样本中，检测为阳性（假阳性）所占的比例**；

**第二类错误率 / 假阴性率：**$\beta = \dfrac{\text{FN}}{\text{TP} + \text{FN}}$，满足$\text{Sn} = 1 - \beta$，**真正阳性样本中，检测为阴性（假阴性）所占的比例**；

**准确率 / 正确率（accuracy）：**$\text{ACC} = \dfrac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{TN}}$，**所有样本中正确检测的比例**；

**召回率：**$\text{Rec} = \dfrac{\text{TP}}{\text{TP} + \text{FN}}$，等价于灵敏度，即**真正阳性样本中，检测为阳性所占的比例**；

**精确率（precision）：**$\text{Pre} = \dfrac{\text{TP}}{\text{TP} + \text{FP}}$，**检测为阳性样本中，真正阳性所占的比例**；

**F度量：**$\text{F} = \dfrac{2\text{Rec}\cdot \text{Pre}}{\text{Rec} + \text{Pre}}$，一种综合评价；

整理：
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" colspan="2" rowspan="2"></th>
    <th class="tg-9wq8" colspan="2">分母：实际为</th>
    <th class="tg-9wq8" colspan="2">分母：检测为</th>
  </tr>
  <tr>
    <th class="tg-9wq8">阳性</th>
    <th class="tg-9wq8">阴性</th>
    <th class="tg-9wq8">阳性</th>
    <th class="tg-9wq8">阴性</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">分子：实际为</td>
    <td class="tg-9wq8">阳性</td>
    <td class="tg-9wq8" colspan="2" rowspan="2">/</td>
    <td class="tg-9wq8">准确率Pre</td>
    <td class="tg-9wq8">/</td>
  </tr>
  <tr>
    <td class="tg-9wq8">阴性</td>
    <td class="tg-9wq8">/</td>
    <td class="tg-9wq8">/</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">分子：检测为</td>
    <td class="tg-9wq8">阳性</td>
    <td class="tg-9wq8">灵敏度Sn / 召回率Rec</td>
    <td class="tg-9wq8">第一类错误率α / 假阳性率</td>
    <td class="tg-9wq8" colspan="2" rowspan="2">/</td>
  </tr>
  <tr>
    <td class="tg-9wq8">阴性</td>
    <td class="tg-9wq8">第二类错误率β / 假阴性率</td>
    <td class="tg-9wq8">特异度Sp</td>
  </tr>
</tbody>
</table>

### 使用Lagrange乘子法求解有约束极值问题
例子：固定第二类错误率，尽量降低第一类错误率
$$\begin{aligned}
  \min \quad& P_1(e) \\
   s.t.\quad& P_2(e) - \varepsilon_0 = 0
\end{aligned}$$

使用Lagrange乘子$\lambda$转化为以下问题：
$$\min\ \gamma = P_1(e) + \lambda(P_2(e) - \varepsilon_0)$$

设$R_1, R_2$分别为两类的决策区域，$R$是整个特征空间，$R_1 + R_2 = R$，两个决策区域之间的边界称作**决策边界**或**分界面（点）**$t$。由概率密度函数性质，有$$\int_{R_2} p(\bm{x} | \omega_1)\,d\bm{x} = 1 - \int_{R_1} p(\bm{x} | \omega_1)\,d\bm{x}$$

代入得$$\begin{aligned}
  \gamma =& \int_{R_2} p(\bm{x} | \omega_1)\,d\bm{x} + \lambda\left[ \int_{R_1} p(\bm{x} | \omega_1)\,d\bm{x} - \varepsilon_0 \right] \\
  =& (1 - \lambda\varepsilon_0) + \int_{R_1} [\lambda p(\bm{x} | \omega_2) - p(\bm{x} | \omega_1)]\,d\bm{x}
\end{aligned}$$

$\gamma$对$\lambda$和$t$求导均为零，则在决策边界上满足$$\lambda = \dfrac{p(\bm{x} | \omega_1)}{p(\bm{x} | \omega_2)}$$同时决策边界使得$$\int_{R_1} p(\bm{x} | \omega_2)\, d\bm{x} = \varepsilon_0$$

又在$\gamma$的表达式中，为了使得$\gamma$最小，我们希望选择$R_1$使得积分项内全部为负值（否则可通过把这部分非负的区域划出$R_1$而使得$\gamma$更小），故$R_1$应该为所有使得$$\lambda p(\bm{x} | \omega_2) - p(\bm{x} | \omega_1) < 0$$成立所构成的区域。

> 从而决策规则为：
> 对某一（接近$0$的）水平$\varepsilon_0$，如果满足$$l(\bm{x}) = \dfrac{p(\bm{x} | \omega_1)}{p(\bm{x} | \omega_2)} \lessgtr \lambda$$则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$；其中$\lambda = \dfrac{p(\bm{x} | \omega_1)}{p(\bm{x} | \omega_2)}$为使决策区域满足$\int_{R_1} p(\bm{x} | \omega_2)\, d\bm{x} = \varepsilon_0$（实际为第二类，被决策为第一类）的阈值。
> 这种限定一类错误率为常数而使得另一类错误率最小的决策规则称为**Neyman-Pearson决策规则**。

由于$\lambda$不方便直接求解，可以用似然比密度函数方法：对于似然比$l(\bm{x}) = \dfrac{p(\bm{x}|\omega_1)}{p(\bm{x}|\omega_2)}$，可知**似然比密度函数**为$p(l | \omega_2)$，$\int_{R_1} p(\bm{x} | \omega_2)\, d\bm{x} = \varepsilon_0$可变为$$P_2(e) = 1 - \int_0^{\lambda} p(l | \omega_2)\, dl = \varepsilon_0$$

### ROC曲线
以假阳性率$1 - \text{Sp}$（即第一类错误率$P_1(e)$）作为横坐标，灵敏度$\text{Sn}$（即真阳性率$1 - P_2(e)$）作为纵坐标，即可得到**ROC曲线**；
左下角$(0, 0)$表示把所有样本都判断为阴性，则假阳性率和真阳性率均为$0$，右上角同理。

**AUC**（area under ROC curves）表示曲线下方相对面积，对角线的AUC为$0.5$，AUC越大方法性能越好。

## 正态分布时的统计决策
### 正态分布
**单变量正态分布：**
$$\mathcal{N}(\mu, \sigma^2): p(x) = \dfrac{1}{\sqrt{2 \pi} \sigma} \exp\left( -\dfrac{1}{2}\left( \dfrac{x - \mu}{\sigma} \right)^2 \right)$$

**多变量正态分布：**
$$\mathcal{N}(\bm{\mu}, \bm{\Sigma}): p(x) = \dfrac{1}{(2 \pi)^{\frac{d}{2}}|\bm{\Sigma}|^{\frac{1}{2}}} \exp \left( -\dfrac{1}{2}(\bm{x} - \bm{\mu})^T \bm{\Sigma}^{-1} (\bm{x} - \bm{\mu}) \right)$$

其中$\bm{x} = (x_1, x_2, ..., x_d)^T$为$d$维列向量；
$\bm{\mu} = (\mu_1, \mu_2, ..., \mu_d)^T$为$d$维列均值向量，满足$E[\bm{x}] = \bm{\mu}$；
$\bm{\Sigma}$为$d \times d$的协方差矩阵，$\bm{\Sigma}^{-1}$为其逆矩阵，$|\bm{\Sigma}|$为其行列式，满足$E[(\bm{x} - \bm{\mu})^T (\bm{x} - \bm{\mu})] = \bm{\Sigma}$。

**性质：**
+ 参数$\bm{\mu}$和$\bm{\Sigma}$对分布的决定性；
+ 等密度点的轨迹为一超椭球面；
+ 不相关性等价于独立性；
    + 两个变量不相关（$E[x_i, x_j] = E[x_i]E[x_j]$）当且仅当两个变量独立（$p(x_i, x_j) = p(x_i) p(x_j)$）
+ 边缘分布和条件分布的正态性；
    + 多元正态分布的边缘分布仍为正态分布；特别地，对于分布为$\mathcal{N}(\bm{\mu}, \bm{\Sigma})$的$d$维多元正态分布，其中第$i\ (i = 1, 2, ..., d)$个变量$x_i$的正态分布为$\mathcal{N}(\mu_i, \sigma_{ii}^2)$（$\sigma_{ii}^2$为$\bm{\Sigma}$对角线上第$i$个元素）。
    + 给定某变量，剩余变量构成的条件分布仍为正态分布。
+ 线性变换的正态性；
    + $\bm{x} \sim \mathcal{N}(\bm{\mu}, \bm{\Sigma})$，若$\bm{y} = \bm{A}\bm{x}$，则$p(\bm{y}) \sim \mathcal{N}(\bm{A\mu}, \bm{A\Sigma A}^T)$
+ 线性组合的正态性；
    + $\bm{x} \sim \mathcal{N}(\bm{\mu}, \bm{\Sigma})$，若$y = \bm{\alpha}^T\bm{x}$（$\bm{\alpha}$为和$\bm{x}$相同维度的列向量），则
    $y \sim \mathcal{N}(\bm{\alpha}^T\bm{\mu}, \bm{\alpha}^T \bm{\Sigma\alpha})$

### 正态分布概率模型下的最小错误率贝叶斯决策
对多元正态概率型$p(\bm{x} | \omega_i) \sim \mathcal{N}(\bm{\mu}_i, \bm{\Sigma}_i),\ i = 1, 2, ..., c$
判别函数：$$g_i(\bm{x}) = -\dfrac{1}{2}(\bm{x} - \bm{\mu}_i)^T \bm{\Sigma}_i^{-1} (\bm{x} - \bm{\mu}_i) - \dfrac{d}{2}\ln 2\pi - \dfrac{1}{2}\ln |\bm{\Sigma}_i| + \ln P(\omega_i)$$

$- \dfrac{d}{2}\ln 2\pi$项和$i$无关，可忽略，从而得到
$$g_i(\bm{x}) = -\dfrac{1}{2}(\bm{x} - \bm{\mu}_i)^T \bm{\Sigma}_i^{-1} (\bm{x} - \bm{\mu}_i) - \dfrac{1}{2}\ln |\bm{\Sigma}_i| + \ln P(\omega_i)$$

决策面方程：$g_i(\bm{x}) = g_j(\bm{x})$，即$$\begin{aligned}
  0 =& -\dfrac{1}{2}\left((\bm{x} - \bm{\mu}_i)^T \bm{\Sigma}_i^{-1} (\bm{x} - \bm{\mu}_i) - (\bm{x} - \bm{\mu}_i)^T \bm{\Sigma}_i^{-1} (\bm{x} - \bm{\mu}_i)\right)\\
  &- \dfrac{1}{2}\ln \dfrac{|\bm{\Sigma}_i|}{|\bm{\Sigma}_j|} + \ln \dfrac{P(\omega_i)}{P(\omega_j)}
\end{aligned}$$

#### 情况一：协方差矩阵均为单位阵的同一倍数矩阵
**即：** $\bm{\Sigma}_i = \sigma^2 I,\ i = 1, 2, ..., c$

+ 判别函数：$$g_i(\bm{x}) = -\dfrac{1}{2 \sigma^2}(\bm{x} - \bm{\mu}_i)^T (\bm{x} - \bm{\mu}_i) + \ln P(\omega_i)$$其中$(\bm{x} - \bm{\mu}_i)^T (\bm{x} - \bm{\mu}_i) = \parallel\bm{x} - \bm{\mu}_i\parallel^2 = \sum\limits_{j = 1}^d (x_j - \mu_{ij})^2,\ i = 1, 2, ..., c$为$\bm{x}$到类$\omega_i$的均值向量$\bm{\mu}_i$的欧氏距离的平方。
+ 由于$\bm{x}^T \bm{x}$和$i$无关，因此可以忽略，判别函数变为$$g_i(\bm{x}) = \dfrac{1}{\sigma^2} \bm{\mu}_i^T \bm{x} + (-\dfrac{1}{2\sigma^2}\bm{\mu}_i^T \bm{\mu}_i + \ln P(\omega_i))$$
    + 可简记为$$g_i(\bm{x}) = \bm{w}_i^T \bm{x} + \omega_{i0}$$其中$\begin{cases}
      \bm{w}_i = \dfrac{1}{\sigma^2} \bm{\mu}_i\\ \omega_{i0} = -\dfrac{1}{2\sigma^2}\bm{\mu}_i^T \bm{\mu}_i + \ln P(\omega_i)
    \end{cases}$
+ 决策规则为对某个待分类的$\bm{x}$，分别计算$g_i(\bm{x}),\ i = 1, 2, ..., c$，取$\omega_k$满足$$k = \argmax_{i} g_i(\bm{x})$$作为最终分类。这是一种**线性分类器**。
+ 决策面由线性方程$g_i(\bm{x}) - g_j(\bm{x}) = 0$决定，该决策面为一个超平面。

#### 情况二：协方差矩阵全部相等
**即：** $\bm{\Sigma}_i = \bm{\Sigma},\ i = 1, 2, ..., c$

+ 判别函数：$$g_i(\bm{x}) = -\dfrac{1}{2}(\bm{x} - \bm{\mu}_i)^T \bm{\Sigma}^{-1} (\bm{x} - \bm{\mu}_i) + \ln P(\omega_i)$$
+ 如果先验概率全部相等，则判别函数可进一步简化为$$g_i(\bm{x}) = \gamma^2 = \dfrac{1}{2}(\bm{x} - \bm{\mu}_i)^T \bm{\Sigma}^{-1} (\bm{x} - \bm{\mu}_i)$$判别规则为：计算$\bm{x}$到每类均值点$\bm{\mu}_i$的Mahalanobis距离平方$\gamma^2$，最后把$\bm{x}$归于$\gamma^2$最小的类别。
+ 由于$\bm{x}^T \bm{\Sigma}^{-1} \bm{x}$和$i$无关，因此可以忽略，判别函数变为$$g_i(\bm{x}) = \bm{\Sigma}^{-1} \bm{\mu}_i^T \bm{x} + (-\dfrac{1}{2}\bm{\mu}_i^T \bm{\Sigma}^{-1} \bm{\mu}_i + \ln P(\omega_i))$$
    + 可简记为$$g_i(\bm{x}) = \bm{w}_i^T \bm{x} + \omega_{i0}$$其中$\begin{cases}
      \bm{w}_i = \bm{\Sigma}^{-1} \bm{\mu}_i\\ \omega_{i0} = -\dfrac{1}{2\sigma^2}\bm{\mu}_i^T \bm{\Sigma}^{-1} \bm{\mu}_i + \ln P(\omega_i)
    \end{cases}$
    + 可见这仍是一种**线性分类器**。
+ 决策面由线性方程$g_i(\bm{x}) - g_j(\bm{x}) = 0$决定，该决策面仍为一个超平面。

#### 情况三：协方差矩阵各不相等
+ 判别函数和决策面见本节开头；
+ 决策面为一个超二次曲面。

## 错误率的计算
在二分类问题中，最小错误率贝叶斯决策的错误率为
$$\begin{aligned}
  P(e) =& P(\omega_1) \int_{\mathscr{R}_2} p(\bm{x} | \omega_1) \,d\bm{x} + P(\omega_2) \int_{\mathscr{R}_1} p(\bm{x} | \omega_2) \,d\bm{x} \\
  =& P(\omega_1)P_1(e) + P(\omega_2)P_2(e)
\end{aligned}$$

当$\bm{x}$为多维向量时，实际上需要进行多重积分的计算。实际应用中，对错误率的计算和估计常分为三种：(1)按理论公式计算；(2)计算错误率上界；(3)实验估计。

### 正态分布且各类协方差矩阵相等情况下错误率的计算
回顾最小错误率贝叶斯决策规则的**负对数似然形式**：
如果满足$$h(\bm{x}) = -\ln [l(\bm{x})] = -\ln p(\bm{x}|\omega_1) + \ln p(\bm{x}|\omega_2) \lessgtr \ln \dfrac{P(\omega_1)}{P(\omega_2)}$$则$\bm{x} \in \begin{cases} \omega_1 \\ \omega_2 \end{cases}$；

记随机变量$h(\bm{x})$的分布函数为$p(h | \omega_1)$，这是一个一维函数，易于积分，从而$$\begin{aligned}
  P_1(e) =& \int_{\mathscr{R}_2} p(\bm{x} | \omega_1) \,d\bm{x} = \int_{t}^{\infty} p(h | \omega_1) \,dh \\
  P_2(e) =& \int_{\mathscr{R}_1} p(\bm{x} | \omega_2) \,d\bm{x} = \int_{-\infty}^{t} p(h | \omega_2) \,dh \\
\end{aligned}$$其中$$t = \ln \dfrac{P(\omega_1)}{P(\omega_2)}$$

对于等协方差矩阵的正态分布情况，有
$$\begin{aligned}
  h(\bm{x}) =& -\ln [l(\bm{x})] = -\ln p(\bm{x}|\omega_1) + \ln p(\bm{x}|\omega_2) \\
  =& -\left( -\dfrac{1}{2} (\bm{x} - \bm{\mu}_1)^T \bm{\Sigma}^{-1} (\bm{x} - \bm{\mu}_1) - \dfrac{d}{2} \ln 2\pi - \dfrac{1}{2} \ln|\bm{\Sigma}| \right) \\
  & + \left( -\dfrac{1}{2} (\bm{x} - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{x} - \bm{\mu}_2) - \dfrac{d}{2} \ln 2\pi - \dfrac{1}{2} \ln|\bm{\Sigma}| \right) \\
  =& (\bm{\mu}_2 - \bm{\mu}_1)^T \bm{\Sigma}^{-1} \bm{x} + \dfrac{1}{2} (\bm{\mu}_1^T \bm{\Sigma}^{-1} \bm{\mu}_1 - \bm{\mu}_2^T \bm{\Sigma}^{-1} \bm{\mu}_2) \\
\end{aligned}$$

该$h(\bm{x})$可以被看成对$\bm{x}$的线性函数，其中对$\bm{x}$的各分量进行线性组合$\bm{\alpha}^T\bm{x}$（其中$\bm{\alpha}^T = (\bm{\mu}_2 - \bm{\mu}_1)^T \bm{\Sigma}^{-1}$）再进行平移。由于$h(\bm{x})$为$\bm{x}$的线性变换，可知$h(\bm{x})$也服从一维正态分布。
对于$p(h|\omega_1)$，可以计算出其一维正态分布的均值$\eta_1$和方差$\sigma_1^2$：
$$\begin{aligned}
  \eta_1 =& \bm{\alpha}^T \bm{\mu}_1 + \dfrac{1}{2} (\bm{\mu}_1^T \bm{\Sigma}^{-1} \bm{\mu}_1 - \bm{\mu}_2^T \bm{\Sigma}^{-1} \bm{\mu}_2)\\
  =& (\bm{\mu}_2 - \bm{\mu}_1)^T \bm{\Sigma}^{-1} \bm{\mu}_1 + \dfrac{1}{2} (\bm{\mu}_1^T \bm{\Sigma}^{-1} \bm{\mu}_1 - \bm{\mu}_2^T \bm{\Sigma}^{-1} \bm{\mu}_2)\\
  =& -\dfrac{1}{2} (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2) \\  
\end{aligned}$$

记$$\eta = -\eta_1 = \dfrac{1}{2} (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2)$$则$$\sigma_1^2 = \bm{\alpha}^T \bm{\Sigma} \bm{\alpha} = (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2) = 2\eta$$

同理，可以求得$p(h|\omega_1)$的一维正态分布均值$\eta_2$和方差$\sigma_2^2$：
$$\begin{gathered}
  \eta_2 = \dfrac{1}{2} (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2) = \eta \\
  \sigma_2^2 = (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2) = 2\eta = \sigma_1^2 = \sigma^2
\end{gathered}$$

从而$$\begin{gathered}
  p(h | \omega_1) \sim \mathcal{N}(-\eta, \sigma^2) = \mathcal{N}(-\dfrac{1}{2} (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2), (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2)) \\
  p(h | \omega_2) \sim \mathcal{N}(\eta, \sigma^2) = \mathcal{N}(\dfrac{1}{2} (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2), (\bm{\mu}_1 - \bm{\mu}_2)^T \bm{\Sigma}^{-1} (\bm{\mu}_1 - \bm{\mu}_2))\\
\end{gathered}$$

利用$p(h | \omega_1)$和$p(h | \omega_2)$计算出$P_1(e)$和$P_2(e)$：
$$\begin{aligned}
  P_1(e) =& \int_{t}^{\infty} p(h | \omega_1) \,dh \\
  =& \int_{t}^{\infty} \dfrac{1}{\sqrt{2\pi} \sigma} \exp\left(-\dfrac{1}{2} \left(\dfrac{h + \eta}{\sigma}\right)^2\right) \,dh\\
  =& \int_{t}^{\infty} \dfrac{1}{\sqrt{2\pi}} \exp\left(-\dfrac{1}{2} \left(\dfrac{h + \eta}{\sigma}\right)^2\right) \,d\left(\dfrac{h + \eta}{\sigma}\right)\\
  =& \int_{\frac{t + \eta}{\sigma}}^{\infty} \dfrac{1}{\sqrt{2\pi}} \exp\left(-\dfrac{1}{2}\xi^2\right) \,d\xi \\
  P_2(e) =& \int_{-\infty}^{t} p(h | \omega_2) \,dh \\
  =& \int_{-\infty}^{\frac{t + \eta}{\sigma}} \dfrac{1}{\sqrt{2\pi}} \exp\left(-\dfrac{1}{2}\xi^2\right) \,d\xi \\
\end{aligned}$$

其中$t = \ln\dfrac{P(\omega_1)}{P(\omega_2)},\ \sigma = \sqrt{2\eta}$。具体计算值可以查标准正态分布的累积分布函数表得到。

### 高维独立随机变量时错误率的估计
当$d$维随机向量$\bm{x}$的分量间相互独立时，$\bm{x}$的密度函数可以表示为$$p(\bm{w} | \omega_i) = \prod_{l = 1}^d p(x_l | \omega_i),\quad i = 1, 2$$

从而负对数似然比$$h(\bm{x}) = \sum_{l = 1}^d h(x_l) = -\sum_{l = 1}^d \ln\dfrac{p(x_l | \omega_1)}{p(x_l | \omega_2)}$$

即$h(\bm{x})$为$d$个随机变量$h(x_l)$的和，根据中心极限定理，当$d$**充分大**时，$h(\bm{x})$的密度函数趋向于正态分布，从而有正态分布参数
$$\begin{aligned}
  \eta_i =& E[h(\bm{x}) | \omega_i] = E\left[\sum_{l = 1}^d h(x_l) | \omega_i\right] = \sum_{l = 1}^d \eta_{il} \\
  \sigma_i^2 =& E[(h(\bm{x}) - \eta_i)^2 | \omega_i] \\
  =& E[\sum_{l = 1}^d (h(x_l) - \eta_{il})^2 + \sum_{l, j = 1 \atop l \neq j}^d (h(x_l) - \eta_{il})(h(x_j) - \eta_{ij}) | \omega_i] \\
  =& E[\sum_{l = 1}^d (h(x_l) - \eta_{il})^2 | \omega_i] + E[\sum_{l, j = 1 \atop l \neq j}^d (h(x_l) - \eta_{il})(h(x_j) - \eta_{ij}) | \omega_i] \\
\end{aligned}$$

其中$\eta_{il}$为属于第$i$类的随机分量$x_l$的期望；由独立性，后一项（即$l \neq j$时）为$0$，从而
$$\sigma_i^2 = E[\sum_{l = 1}^d (h(x_l) - \eta_{il})^2 | \omega_i] + 0 = \sum_{l = 1}^d \sigma_{il}^2$$
