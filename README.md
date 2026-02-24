# Sin 外推能力测试

这个实验用 `sin(x)` 在 `[0, 2π]` 区间内的随机采样点作为训练集，测试不同模型在 `[2π, 4π]` 区间的外推能力。

## 实验设置
- 训练数据：`x ~ Uniform(0, 2π)`，随机不等间距采样。
- 目标函数：`y = sin(x)`。
- 评估区间：`[2π, 4π]`（纯外推）。
- 比较模型（纯 Python 实现）：
  - 简单 MLP（1 hidden layer, tanh）
  - Random Forest（1D CART + bagging）
  - KNN（距离加权回归）
- 比较训练样本数：`20, 50, 120`
- 指标：外推区间 RMSE（越小越好）。

## 运行方式
```bash
python extrapolation_experiment.py
```

运行后会生成：
- `results/sin_extrapolation_comparison.svg`：不同模型在不同样本量下的拟合/外推曲线。
- `results/metrics.csv`：外推区间 RMSE 汇总表。

## 可选扩展
你可以在 `extrapolation_experiment.py` 中：
- 修改 `sample_sizes` 比较更多样本规模。
- 在 `models` 列表里增加新的模型实现进行对比。
- 在训练数据里加入噪声，测试鲁棒性。
