Run BP.m
神经网络输出值具有一定的波动性，每次的输出值都会在一定范围内波动，而自己仅需该范围内某个特定小范围值。此时可以借助下列模板进行获取：
```
while 1==1
    神经网络计算得到的函数值（Y）
    if Y==0.5
    break
    end
end
```
博客连接：https://juejin.cn/post/7073774406173982756
