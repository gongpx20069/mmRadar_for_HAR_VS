### 1. 数据体素化（可能弃用）

> 体素化是将物体几何形式表示转换成最接近物体的体素表示形式，产生体数据集，其不仅包含模型的表面信息，而且能描述模型的内部属性。表示模型的空间体素与表示图像的二维体素比较相似，只不过从二维的点扩展到三维的立方体单元，而且基于体素的三维模型有很多应用。

#### 1.1 数据体素化表示流程

由于使用的需要，需对模型进行体素化操作，这里采用了一种简单但却有效的操作方法。**首先设定模型体素化的分辨率，假设为$N \times N \times N$**，之后的操作则主要包含两部分：

- 对模型表面体素化
- 对模型内部体素化

#### 1.2 空间点的体素化

假设点位置为$(X, Y, Z)$，体素坐标系是一种局部坐标系，其X，Y，Z轴与输入点云数据的X，Y，Z轴方向相同，坐标原点与原始点云数据中三个坐标的最小值构成的点即$(X_{min}, Y_{min}, Z_{min})$相对应，**其坐标均为不小于0的整数，表示体素在该坐标轴方向中的位置。**
$$
f(x)=\left\{
\begin{aligned}
x  =  N \times INT((x-x_{min})/(x_{max}-x_{min})) \\
y  =  N \times INT((y-y_{min})/(y_{max}-y_{min})) \\
z  =  N \times INT((z-z_{min})/({z_{max}-z_{min}}))
\end{aligned}
\right.
$$

### 2. 雷达选型

相关代码包：[ti mmwave rospkg](https://github.com/radar-lab/ti_mmwave_rospkg)

#### 2.1 TI IWR1443 FMCW mmWave radars

德州仪器IWR1443毫米波雷达（工业级器件）

#### 2.2 TI AWR1443 mmWave radars

德州仪器IWR1443毫米波雷达（汽车级器件）

### 3. 相关工作

#### 3.1 PointNet

点云一般有如下特征：

- 无序性：在几何上，点的顺序不影响它在空间中对整体形状的表示，例如，相同的点云可以由两个完全不同的矩阵表示。比如下图：
  $$
  \left[ 
  \begin{matrix} 
  0 & 1 & 2 \\ 
  3 & 4 & 5 \\ 
  \end{matrix} 
  \right] 
  \not=
  \left[ 
  \begin{matrix} 
  3 & 4 & 5 \\ 
  0 & 1 & 2 \\ 
  \end{matrix} 
  \right]
  $$

- 旋转性：相同的点云在空间中经过一定的刚性变化（旋转或平移），坐标发生变化，但实际上其本质未变；
- 相互作用：点是来自一个有距离度量的空间，这意味着点云并不是孤立的，相邻的点形成一个有意义的子集。

PointNet核心思想：

- 使用对称函数解决点云无序性的问题：
  $$
  f({x_1,...,x_n}) \approx g(h(x_1),...,h(x_n))
  $$

其中，$h$为简单神经网络（线性变换+点卷积+MLP），g为对称函数，可以用MaxPool、MeanPool、SumPool等操作，在论文中使用的是Maxpool。

![image-20201112170504032](md_img\image-20201112170504032.png)

#### 3.2 PointNet++

在PointNet++是PointNet的扩展，为弥补PointNet没有考虑**点云相互作用**的的问题，因此提出的PointNet++。

PointNet++如下结构：

![image-20201112171139400](md_img\image-20201112171139400.png)

PointNet架构主要是set abstraction操作，而这个操作分为三步：

- Sampling：这个操作主要是选取**局部区域的中心**，采用的算法为FPS(Farthest Point Sampling)，这种采样方法可以尽可能覆盖空间中的点，详情可点[此处](https://blog.csdn.net/QFJIZHI/article/details/103419044)。这一层网络的输入为$(N,d+C)$，其中N为点云所有点的个数，d为该点的位置，C为该点特征。在这个阶段，我们可以得到$N'$个中心点（人为定义$N'$）。
- Grouping：在这个阶段，我们需要使用KNN算法或者欧氏距离画圆进行多个group的点的划分，在上一步我们得到了$N'$个中心点，假设每个group的点个数为K（KNN算法），那么最终输出的数据格式为$(N',K,d+C')$，其中，d为各点的位置，C'为各点特征。
- pointnet：如上一章节$f({x_1,...,x_n}) \approx g(h(x_1),...,h(x_n))$



#### 3.3 Transformer



### 4. 代码

#### 4.1 HAR/code_v1

第一版代码主要来源于论文《RadHAR: Human Activity Recognition from Point Clouds Generated through a Millimeter-wave Radar》 在其github代码上修改而来的TD_CNN_LSTM网络（Pytorch版本） 代码地址为：https://github.com/nesl/RadHAR

在我们的测试中，最终结果在第12个epoch达到了86%，一定程度上可以说明该网络的有效性：

```python
Test Accuracy 20.6049%
epoch:0	 epoch loss:2497.9695
Test Accuracy 59.7795%
epoch:1	 epoch loss:1980.7943
Test Accuracy 78.1798%
epoch:2	 epoch loss:1769.9320
Test Accuracy 82.9565%
epoch:3	 epoch loss:1685.8812
Test Accuracy 85.8960%
epoch:4	 epoch loss:1643.4520
Test Accuracy 87.0548%
epoch:5	 epoch loss:1618.5165
Test Accuracy 86.2634%
epoch:6	 epoch loss:1605.9521
Test Accuracy 86.0090%
epoch:7	 epoch loss:1597.8629
Test Accuracy 84.5393%
epoch:8	 epoch loss:1590.9446
Test Accuracy 87.3940%
epoch:9	 epoch loss:1584.7410
Test Accuracy 87.5636%
epoch:10	 epoch loss:1582.8142
Test Accuracy 85.4720%
epoch:11	 epoch loss:1579.6831
Test Accuracy 84.8785%
epoch:12	 epoch loss:1575.7629
Test Accuracy 86.9700%
```

这一版的代码使用了体素化表示点云，当前版本代码存在问题：

- 体素化表示点云处理过程复杂，稀疏点占用空间巨大，无法实现端到端；

- 没有显式的局部信息获取；
- 点云本质具有set的性质，本身是无序信息，而图像（立体场景）是有序信息；
- 速度信息未考虑。

