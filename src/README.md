# Python 源码

`msabp_opt` 是项目的可导入核心包。依赖方向建议保持为：

```text
geometry → simulation → postprocessing
        ↘ optimization ↗
```

流程入口应调用这里的模块，而不是从另一个脚本复制函数。
