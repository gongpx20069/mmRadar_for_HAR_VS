### 2. 代码

#### 1.1 HAR/code_lstm

这一版的代码主要是作为对比，由于我们之后的论文基本都用了LSTM，因此我们计划是使用相同的配置进行最终效果的对比（尤其是和PointGNN）

最终测试的结果为：

```python
Test Accuracy 23.8630%
epoch:1	 epoch loss:189.8299
Test Accuracy 35.6822%
epoch:2	 epoch loss:178.9261
Test Accuracy 41.6901%
epoch:3	 epoch loss:173.0116
Test Accuracy 43.5149%
epoch:4	 epoch loss:170.0830
Test Accuracy 43.8237%
epoch:5	 epoch loss:168.5532
Test Accuracy 44.1044%
epoch:6	 epoch loss:167.7885
Test Accuracy 44.2448%
epoch:7	 epoch loss:167.3001
Test Accuracy 44.4413%
epoch:8	 epoch loss:167.0112
Test Accuracy 44.4975%
epoch:9	 epoch loss:166.8356
Test Accuracy 44.5817%
epoch:10	 epoch loss:166.7532
Test Accuracy 44.6098%
epoch:11	 epoch loss:166.7044
Test Accuracy 44.6098%
epoch:12	 epoch loss:166.6888
Test Accuracy 44.6098%
epoch:13	 epoch loss:166.5931
Test Accuracy 44.5817%
epoch:14	 epoch loss:166.5834
Test Accuracy 44.5817%
epoch:15	 epoch loss:166.5637
```



#### 1.2 HAR/code_v1(TDCNN+LSTM)

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



#### 1.3 HAR/code_v2(TDPointNet+LSTM)

这一版的代码主要使用了PointNet和STN以及BLSTM组合的结构，没有使用点云体素化，而是使用了(batch_size, 60, 42, 3)这样**占用空间很小**，处理很快的点云集合。最终得到的效果如下图（前50个epoch的学习率为0.00001，51-72的学习率为0.000001），**这个版本的代码已经超过了RadHAR论文中90%的最高准确率**：

```python
epoch:50	 epoch loss:137.8508
Test Accuracy 91.8024%
epoch:51	 epoch loss:137.8264
Test Accuracy 91.7462%
epoch:52	 epoch loss:137.8304
Test Accuracy 92.0270%
epoch:53	 epoch loss:137.8096
Test Accuracy 92.7007%
epoch:54	 epoch loss:137.7618
Test Accuracy 91.9708%
epoch:55	 epoch loss:137.7401
Test Accuracy 92.1112%
epoch:56	 epoch loss:137.7477
Test Accuracy 92.0550%
epoch:57	 epoch loss:137.7485
Test Accuracy 91.1286%
epoch:58	 epoch loss:137.7491
Test Accuracy 91.2970%
epoch:59	 epoch loss:137.7663
Test Accuracy 91.6901%
epoch:60	 epoch loss:137.7136
Test Accuracy 92.1393%
epoch:61	 epoch loss:137.7390
Test Accuracy 91.4655%
epoch:62	 epoch loss:137.7053
Test Accuracy 91.3812%
epoch:63	 epoch loss:137.7313
Test Accuracy 90.5671%
epoch:64	 epoch loss:137.7269
Test Accuracy 91.7743%
epoch:65	 epoch loss:137.7476
Test Accuracy 91.2970%
epoch:66	 epoch loss:137.7277
Test Accuracy 91.0444%
epoch:67	 epoch loss:137.8045
Test Accuracy 91.6901%
epoch:68	 epoch loss:137.7187
Test Accuracy 92.3077%
epoch:69	 epoch loss:137.7257
Test Accuracy 93.3745%
epoch:70	 epoch loss:137.7416
Test Accuracy 92.0831%
epoch:71	 epoch loss:137.6977
Test Accuracy 91.4935%
epoch:72	 epoch loss:137.7179
Test Accuracy 92.0550%
epoch:73	 epoch loss:137.6846
Test Accuracy 92.1393%
epoch:74	 epoch loss:137.7253
Test Accuracy 91.4935%
epoch:75	 epoch loss:137.6949
Test Accuracy 92.1393%
epoch:76	 epoch loss:137.6889
Test Accuracy 92.3077%
epoch:77	 epoch loss:137.6856
Test Accuracy 91.5497%
epoch:78	 epoch loss:137.7148
Test Accuracy 92.3077%
```

#### 1.4 HAR/code_v3(TDPointGNN+LSTM)

当前版本使用的框架是PointGNN+LSTM

设置为$T = 3,r=0.05,learning\,rate=0.0001,optim_{LR}=0.6,state_dim=3(xyz)$，每个顶点的平均边为30，结果为：

```python
Test Accuracy 23.7226%
epoch:1	 epoch loss:1657.6987
Test Accuracy 63.1106%
epoch:2	 epoch loss:1362.5378
Test Accuracy 70.9152%
epoch:3	 epoch loss:1307.5026
Test Accuracy 73.3015%
epoch:4	 epoch loss:1286.3761
Test Accuracy 74.7052%
epoch:5	 epoch loss:1276.6403
Test Accuracy 74.3403%
epoch:6	 epoch loss:1270.6652
Test Accuracy 74.8175%
epoch:7	 epoch loss:1267.3583
Test Accuracy 74.5368%
epoch:8	 epoch loss:1265.3301
Test Accuracy 74.8175%
epoch:9	 epoch loss:1264.2133
Test Accuracy 74.9579%
epoch:10	 epoch loss:1263.4835
Test Accuracy 75.0140%
epoch:11	 epoch loss:1263.0441
Test Accuracy 75.0140%
epoch:12	 epoch loss:1262.7979
Test Accuracy 74.9579%

```

设置为$T = 3,r=0.0005,learning\,rate=0.0001,optim_{LR}=0.6,state_dim=3(xyz)$，每个顶点的平均边为10，结果为：

```python
Test Accuracy 20.1011%
epoch:1	 epoch loss:1638.2410
Test Accuracy 56.1202%
epoch:2	 epoch loss:1401.5388
Test Accuracy 70.8591%
epoch:3	 epoch loss:1309.8812
Test Accuracy 72.7400%
epoch:4	 epoch loss:1287.3472
Test Accuracy 73.2734%
epoch:5	 epoch loss:1276.7604
Test Accuracy 73.5542%
epoch:6	 epoch loss:1270.7698
Test Accuracy 73.8630%
epoch:7	 epoch loss:1267.1136
Test Accuracy 74.1718%
epoch:8	 epoch loss:1265.4233
Test Accuracy 73.9472%
epoch:9	 epoch loss:1264.0820
Test Accuracy 74.1437%
epoch:10	 epoch loss:1263.3696
Test Accuracy 74.0034%
epoch:11	 epoch loss:1262.9706
Test Accuracy 74.2560%
epoch:12	 epoch loss:1262.6624
Test Accuracy 74.2560%
epoch:13	 epoch loss:1262.5419
Test Accuracy 74.1999%
epoch:14	 epoch loss:1262.5200
Test Accuracy 74.2280%
epoch:15	 epoch loss:1262.3126
Test Accuracy 74.2280%
epoch:16	 epoch loss:1262.2665
Test Accuracy 74.2280%
epoch:17	 epoch loss:1262.2375
Test Accuracy 74.2280%
epoch:18	 epoch loss:1262.2306
Test Accuracy 74.2280%
epoch:19	 epoch loss:1262.2253
Test Accuracy 74.2280%
epoch:20	 epoch loss:1262.3317
Test Accuracy 74.2280%
epoch:21	 epoch loss:1262.3201
Test Accuracy 74.2280%
epoch:22	 epoch loss:1262.2209
Test Accuracy 74.2280%
epoch:23	 epoch loss:1262.3145
Test Accuracy 74.2280%
epoch:24	 epoch loss:1262.2755
Test Accuracy 74.2280%
epoch:25	 epoch loss:1262.2997
Test Accuracy 74.2280%
epoch:26	 epoch loss:1262.2167
Test Accuracy 74.2280%
epoch:27	 epoch loss:1262.4041
Test Accuracy 74.2280%
epoch:28	 epoch loss:1262.2290
Test Accuracy 74.2280%
epoch:29	 epoch loss:1262.3732
Test Accuracy 74.2280%
epoch:30	 epoch loss:1262.3842
Test Accuracy 74.2280%
epoch:31	 epoch loss:1262.2272
Test Accuracy 74.2280%
```

**这里一定程度上可以说明，顶点邻接点的个数对于最终的结果影响是不大的**

我们将毫米波雷达的所有参数加上，因此state_dim=8（包括该点的位置），$T = 3,r=0.0005,learning\,rate=0.001,optim_{LR}=0.8,state_dim=8$，最终的结果如下：

```python
Test Accuracy 86.8052%
epoch:18         epoch loss:1630.4952    learning rate:0.00040500000000000003
Test Accuracy 88.0685%
epoch:19         epoch loss:1603.5416    learning rate:0.0003645
Test Accuracy 88.0124%
epoch:20         epoch loss:1593.4146    learning rate:0.00032805000000000003
Test Accuracy 88.9669%
epoch:21         epoch loss:1587.6666    learning rate:0.000295245
Test Accuracy 89.8933%
epoch:22         epoch loss:1586.7390    learning rate:0.0002657205
Test Accuracy 90.8478%
epoch:23         epoch loss:1580.0706    learning rate:0.00023914845
Test Accuracy 90.3425%
epoch:24         epoch loss:1581.9740    learning rate:0.000215233605
Test Accuracy 90.3144%
epoch:25         epoch loss:1577.9630    learning rate:0.0001937102445
Test Accuracy 91.4093%
epoch:26         epoch loss:1575.1732    learning rate:0.00017433922005
Test Accuracy 91.9708%
epoch:27         epoch loss:1573.7748    learning rate:0.00015690529804500002
Test Accuracy 91.1286%
epoch:28         epoch loss:1574.0050    learning rate:0.00014121476824050002
Test Accuracy 91.1005%
epoch:29         epoch loss:1573.1562    learning rate:0.00012709329141645002
Test Accuracy 91.0163%
epoch:30         epoch loss:1573.7720    learning rate:0.00011438396227480502
Test Accuracy 90.7636%
epoch:31         epoch loss:1573.1935    learning rate:0.00010294556604732453
Test Accuracy 92.1112%
epoch:32         epoch loss:1572.6886    learning rate:9.265100944259208e-05
Test Accuracy 91.1847%
epoch:33         epoch loss:1572.3966    learning rate:8.338590849833288e-05
Test Accuracy 91.8585%
epoch:34         epoch loss:1572.0251    learning rate:7.50473176484996e-05
Test Accuracy 92.1673%
epoch:35         epoch loss:1571.8701    learning rate:6.754258588364964e-05
Test Accuracy 91.6058%
epoch:36         epoch loss:1571.8706    learning rate:6.078832729528468e-05
Test Accuracy 91.2409%
epoch:37         epoch loss:1571.6724    learning rate:5.4709494565756215e-05
Test Accuracy 91.7181%

```

其中的最高准确率达到了92.167%，我们在此基础上对点云图神经网络进行了修改，将其连接边也进行了更新。代码如HAR/code_v3。

#### 1.5 HAR/code_v4(TDPointGNN_boost+LSTM)

我们修改了PointGNN的更新方式$T = 3,r=5,learning\,rate=0.001,optim_{LR}=0.8,state_dim=8$，在每帧图像都得到[42, 8]个状态值后，我们又对8个状态映射为[42, 128]。取最大后的结果为[128]输入Bi-LSTM。

更新公式如下：
$$
\Delta x_i^t = MLP_{h}^{t}(s_i^t) \\
e_{ij}^t = MLP_f^t([x_j-x_i + \Delta x_i^t, s_j^t]) \\
\Delta A^t = MLP_r^t(e_{ij}^t) \\
A^{(t+1)} = hard\_sigmoid(3*(A^t + \Delta A^t)) \\
s_i^{t+1} = MLP_g^t(Max\{e_{ij}|(i,j) \in E^t\})+s_i^t
$$

在网络推断时，$A^{t+1}$的求取方式更改为：
$$
A^{t+1} =\{A_{ij}^{t+1}=1|A_{ij}^t+\Delta A_{ij}^t > 0\}
$$


最终的结果如下：

```python
epoch:42         epoch loss:2212.4436    learning rate:0.00042805066795449306
Test Accuracy 94.2729%
epoch:43         epoch loss:2208.4946    learning rate:0.0004194896545954032
Test Accuracy 96.9680%
epoch:44         epoch loss:2212.3975    learning rate:0.0004110998615034951
Test Accuracy 95.4239%
epoch:45         epoch loss:2207.8750    learning rate:0.00040287786427342523
Test Accuracy 95.4520%
epoch:46         epoch loss:2208.2219    learning rate:0.00039482030698795675
Test Accuracy 95.6204%
epoch:47         epoch loss:2211.0491    learning rate:0.0003869239008481976
Test Accuracy 94.9467%
epoch:48         epoch loss:2209.0977    learning rate:0.0003791854228312336
Test Accuracy 95.3959%
epoch:49         epoch loss:2212.6357    learning rate:0.00037160171437460894
Test Accuracy 95.6204%
epoch:50         epoch loss:2207.3765    learning rate:0.00036416968008711674
Test Accuracy 95.2836%
epoch:51         epoch loss:2207.7761    learning rate:0.0003568862864853744
Test Accuracy 94.8624%
epoch:52         epoch loss:2207.7495    learning rate:0.0003497485607556669
Test Accuracy 94.6098%
epoch:53         epoch loss:2207.8184    learning rate:0.00034275358954055353
Test Accuracy 95.0028%
epoch:54         epoch loss:2209.1448    learning rate:0.00033589851774974244
Test Accuracy 94.0483%
epoch:55         epoch loss:2207.2847    learning rate:0.0003291805473947476
Test Accuracy 95.1151%
epoch:56         epoch loss:2206.5386    learning rate:0.0003225969364468526
Test Accuracy 94.6378%
```

在MMPoint-GNN中，我们的方法很快达到了96.9680%的准确率，超过了code_v3中的93%，目前是效果最好的网络。
