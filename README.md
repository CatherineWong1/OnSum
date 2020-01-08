# OnSum
##数据集格式：
src, src_txt, target, target_txt, df, label


##整体的思路：
对df的值进行一个二分类，每一个位置上最后的值都是0或1，去和固定的lable做ground truth

###网络结构
Train阶段：
```
Input：df: [batch_size, doc_len]

Architecture： 使用一个NN来训练
binary_output = sigmoid(W * input + b)
binary_output = [batch_size, doc_len]

loss function: BCEloss

lr = 0.00001
optimizer: Adam
```
Test阶段:
```
1. 通过Model得到一个binary_output
2. 通过binary_output, 拼接具体的句子，原来的summary计算roug
```
