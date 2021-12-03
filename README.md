**Topic in Data Science: Spectral Method and Nonconvex optimization**

literature review
- Paper: [Spectral method meet EM](https://www.jmlr.org/papers/volume17/14-511/14-511.pdf)
- [data and matlab implement by author](https://github.com/zhangyuc/SpectralMethodsMeetEM)
- team member: Aoxue Chen, Song Liang
- Other references:
    - [Tensor decompositions for learning latent variable models](https://www.jmlr.org/papers/volume15/anandkumar14b/anandkumar14b.pdf)

Interesting problems that raise along with our implementation:

- [ ] $M_2$和$M_3$是不是对称张量有没有影响？（已在函数中整合是否寻找临近对称张量的参数`sym`）
- [ ] 拟合的confusion matrix是负数的话需不需要在0处截断（已在函数中整合截断选择参数`cutoff`）
- [ ] 除了论文里列出的其他特例
- [ ] 有没有什么步骤是不需要的？
- [ ] 缺失数据的量、数据规模、类别数量等因素对算法是否有影响
- [x] 使用张量操作代替循环以提高算法性能（已完成）
- [x] 在有人没给很多物品打标的情况下效果不是很好（已解决）

