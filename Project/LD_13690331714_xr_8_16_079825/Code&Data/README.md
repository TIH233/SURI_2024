# manual
## Simulation_Optimal
通过搜索算法（scatter search& tabu search)分别评估目标函数，即cost的大小，以此得出最有的解（r,Q）值

### 运行
Jupyter Notebook便于观看，但是运行时会和pool结构冲突，所以建议用.py

.py文件我是拿.ipynb转换来的，所以存有cell的运行按钮。在 if __name__ == '__main__': 位置也有一个entry

跑的时候按那个会运行底下所有部分，按cell的按钮会显示wrapper_objetive_function cannot be accessed
应该是module方面的问题，wrapper函数容易被认为定义在module内（同一个文件没有这个if __name__ == '__main__':来specify的话）

Jupyter Notebook跑之前的问题就是如果并行运行垮了，他还会继续跑，就会开启很多thread但还是单线程运行，memory正常占但CPU就很少（0.4%）
### 内容
前面都是定义函数后面都是实际运行

helper method定义了一个simulation里面记录什么数值，事件设置和顺序。最后log in是用来track process的

Optimizer 定义优化的程序，

1. 两个search function，并在其中包括跑仿真和evaluate，compare的工作

2. 一个仿真跑20000单位时间，一个solution跑五次取均值
3. 如果在生成solution的时候碰到了边界，边界会自动扩大（expand rate),并且会重新用这个解开始跑一遍
4. check_convergence 如果发现未来100个解的结果提升度不超过1%，认定有最优解然后跳出。达到最大scenario的次数（500)也会跳出。（但是我怀疑这部分没包括检索出最优解？可能也没问题毕竟没报错）
5. objective_function_wrapper & SolutionCache： 在每次跑objective function前对照看以前有没有跑过这个解，在一定范围内（tolerance),视作跑过
6. optimize_inventory_system 最终的函数，把所有的过程连在一起

multiprocessing 用在了

1. scatter solution（在optimize_inventory_system里面被call所以他自己定义一个pool）把多个解并行运算
2. run_statistics_collection 把30个rep并行运算

后面parameter初始设下后就开始运行了，跟着就要把随机生成的demand（那个生成函数）记录在.csv和记录statistics到文件里面，并且设了checkpoint[]用来保留进程的但是现阶段都是报错跑不起来所以可以考虑comment

最后部分用最优的解再跑一遍验证之前的普适性(validation)

### 问题和方向
Error1,2 是前几个版本的报错

最初是objective_function不能serialize(pickled),所以用了个wrapper函数去让pool可以有他名字的access

然后就是一个脚本文件被视作一个module，然后并行运算的child process是access不到一个module里面定义的函数，所以用if __name__ == '__main__': 划分了一下应该，同时据说可以保证不会有recursive process发生

之后就是公用pool还是一个pool的问题，是否要定义两个with multiprocessing.Pool(num_processes) as pool:

具体体现就是跑的时候报错statistics变量在用的时候没有被定义（error3文件里），说明当时一个pool的时候就不是sequential地算，optimization, statistics collection可能同时在不同子过程编译器上跑，那又牵扯到变量数据共享问题（不同child process里面变量只有在return to main的时候才会对齐？)

**方向**就是之前提到过一个把optimization和statistics collection函数分别装在不同文件里面，然后在主文件调用，据说在module定义上的access更方便

然后我理想型是这两函数可以分别跑，这样track process, debug也容易，然后中间的参数和跑的次数(maximum_scenario, n_rep, etc)最好不要该因为是文献里做对比组的

但是像expand rate, initial solution这些算法具体实施的参数你跑的怎么优化怎么来

当然你要test的话改肯定可以，我的意思就是确保我这个参数能运行起来（因为之前试了3个场景能跑，50个就不行，可能是并行运行里面变量会堆积冲突之类的问题）

然后希望你最后确保能跑出来的时候跟我说一下计算量大概多少(CPU 100% 跑个半小时我这电脑怕是不行得换别的)


