# xueyeyujing — 学生 CGPA 预测流水线

## 用 PyCharm 运行（推荐）

### 第一步：用 PyCharm 打开项目

1. 打开 PyCharm → **File → Open** → 选择本项目文件夹
2. PyCharm 会自动识别项目（`.idea` 配置已包含在内）

### 第二步：配置 Python 解释器

1. 菜单栏 → **File → Settings → Project: xueyeyujing → Python Interpreter**
2. 点击右上角齿轮图标 → **Add Interpreter → Add Local Interpreter**
3. 选择 **Virtualenv Environment** → 新建虚拟环境，点击 **OK**

### 第三步：安装依赖

在 PyCharm 底部打开 **Terminal**，输入：

```bash
pip install -r requirements.txt
```

### 第四步：一键运行

- 顶部工具栏右侧会出现运行配置 **"运行 CGPA 预测流水线"**
- 点击绿色三角形 ▶ 即可运行

> 也可以直接打开 `model_pipeline.py`，点击右上角绿色三角形运行（所有参数均有默认值，无需手动配置）。

---

## 命令行运行

```bash
python model_pipeline.py --data Students_Performance_dataset.csv --output . --n_iter 50 --random_state 42
```

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `results.csv` | 各模型 R²/RMSE/MAE 对比 |
| `ablation_results.csv` | 消融实验结果 |
| `*.png` | 模型对比图表（4 张） |
| `ablation_r2.png` | 消融实验 R² 对比图 |