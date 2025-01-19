## 反馈
### 方法论

我把源码和文章具体过了一遍，总体来说工作量确实大，我这就给你介绍一下后你看着做

他文章是把原数据集differencing处理

D (t)DailyDiff = D (t)Actual − D (t − 1)Actual

D (t)WeeklyDiff = D (t)Actual − D (t − 7)Actual

三个数据集做相同的处理， performance metrics直接比较

然后先用F-score去給feature的相关性排序

再在模型训练的时候用一个loop分别试number of features 从 1 到 73 (从rank高到低开始选)

然后预测分别预测一天后和一周后，一周后用的的multiple regressor直接求和

最后得出预测什么数据集的多少天后的sales用什么模型最好

### 结果
最终得出， 在预测actual data（原数据集）的一周后销量

LTSM和GRU （都是RNN）比baseline高，而且比使用其他数据集表现都好

### 源码
scikit是 nonRNN model， Keras是RNN model
格式有.py也有Jupyter notebook .ipynb

因为他在一个文件不同cell里面对应不同数据集相同工作，我把预测原数据集一周后的部分提出来了

在NonRNN.ipynb

主体就是import导入包（这里我用的直接就是源码里面的，因为这个适用所有模块）

**helper method定义函数：**


1. 数据流output behavior define (源码是即输出到terminal也写上文件，我改成就写上文件了)

2. Pearson correlation计算

3. Dataset rolling windows计算 （这个函数我没怎么看懂但是用来处理数据集变成时序分析的形式，我觉得可以不用改）

4. 假期栏的onehotencoding
5. 把历史数据加入作为feature用来预测，最终输出modified的数据集

这个可以对照论文上feature的那个表来看

**Model Initialization初始化模型：**

他把所有要测试的模型输入一个预训练的参数（应该是）

**Actual Test真正开始训练和比较：**

里面有具体数据集操作（具体哪些column直接删)，然后一整个流程下来

### 关于我看完后的看法

这个代码理解有点难度，我现在觉得也没必要，也干不了我之前说的那啥自动化，什么自动识别做决策

**就按这个case具体的数据集来 （actual data)，在这个基础上加一格aggregation和选择的模块 (他的调参我看起来就像是自动的）**

你可能需要读RNN那个文件，然后要么跟nonRNN文件关联，要么直接加个模块进去

然后集合的供选择的算法就拿LTSM （加GRU如果你觉得合适的话）和几个表现好的nonRNN （虽然没有baseline好但是集合后这个具体案例只要LTSM好整体就会更好）

然后可能ranking这块可以拿random forecast去分析重要性（F score只能捕捉线性），具体实现就加个模块然后改那个ranking的list，你整个改也没问题

### 总结

在这个基础上你可以加你自己的理解去发挥，改动，

只要最后的结果可以比baseline好。跟这个paper的结果比我觉得不太实际，后面就可以写我们的用别的数据集表现可能更好

因为这个最终的产出就是一水论文，而且审核的也不是专业审稿人，所以我觉得面子工程做好了就可以

比如只要最后的visualization显示不错，具体实现细节我觉得没人会过问

所以我的意思是，你拼接，缝合，抄，GPT，都没问题，只要不太明显 (比如把源码一点不改)，就算跟源码大批相像，只要有创新都能说的过去

CSDN, paper with code我觉得可以去看看。前者都是比较简单的模型（性能可能不会很好），后者可以查到时序分析的advance model但是阅读量又大了，hugging face没有这方向的，kaggle可以往时序预测那块查

看完后你觉得思路有问题找我，觉得没问题但有需要我做的地方也可以找我，总之能水出结国就可以

关于加钱我之前说加1K是底线，现在看来我觉得1K5左右我都能接受，你要觉得不合理也可以找我商量



