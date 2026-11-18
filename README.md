# 金融量化分析与回测平台

基于 Flask 与 ECharts 的股票预测及量化回测实验项目，覆盖行情数据、特征工程、多模型训练、策略回测和网页可视化。

## 功能

- 支持 XGBoost、SVM、BP 神经网络（MLP）和逻辑回归。
- 支持基础指标、增强因子、时序记忆及行业特征的组合与消融实验。
- 对比未来 1 日 / 5 日、绝对涨跌幅 / 横截面相对强弱标签。
- 输出 Accuracy、Precision、Recall、F1、AUC、混淆矩阵及 ROC 曲线。
- 对比含摩擦成本和无摩擦成本的资金曲线，支持仓位、止盈止损与交易阈值配置。
- 使用时序切分评估不同市场状态下的模型表现。

## 目录

| 路径 | 内容 |
| --- | --- |
| `app/` | Flask 路由、HTML 页面、ECharts 图表与样式 |
| `core/` | 特征处理、模型封装、评估与回测逻辑 |
| `scripts/` | 数据抓取、特征构建、训练与回测脚本 |
| `core/config.py` | 特征组、训练测试区间、股票池和回测参数 |
| `main.py` | 本地 Web 启动入口 |
| `README.txt` | 原项目详细说明 |

## 运行

原开发环境为 Python 3.9。建议使用独立 Python 环境，并在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python main.py
```

浏览器访问 `http://127.0.0.1:5000`。原有完整本地项目保留数据和模型时，可以继续使用原来的运行方式。

依赖目前未锁定版本；从其他机器安装最新依赖不保证与历史模型缓存兼容。

## 数据与训练

仓库保留源码；本地行情 CSV、加工特征、序列化模型、实验输出和含个人信息的课程报告未上传，也未从本机删除。仅克隆仓库不能直接完成全部回测。

特征构建需要 `data/raw/stock_daily.csv` 和 `data/raw/index300_daily.csv`，后者需包含代码要求的 `index300_close` 字段。请自行准备有权使用的数据，并遵守数据来源的使用要求。

`scripts/fetch_real_data.py` 使用 Baostock 抓取个股行情；现有脚本保存到 `scripts/data/raw/stock_daily.csv`，而特征脚本读取根目录 `data/raw/stock_daily.csv`，使用时需自行将抓取结果放到正确位置。当前抓取脚本没有补齐沪深 300 数据的流程。

数据就绪后，在项目根目录依次执行：

```powershell
python scripts/step01_build_features.py
python scripts/02_train_models.py
python main.py
```

在网页中选择模型与参数运行回测。训练及回测可能耗时较长，不属于启动首页检查的一部分。

## 提交修改

直接在本项目中修改，然后执行：

```powershell
git status
git add .
git commit -m "说明这次修改"
git push
```

数据、模型、输出和报告已由 `.gitignore` 排除。提交前仍请检查文件列表。

## 验证范围

上传前在原 Python 环境验证了 19 个 Python 文件的语法、首页 HTTP 200，以及回测接口对非 JSON 请求返回 HTTP 400。未重新抓取数据、训练模型或运行完整回测；现有模型效果不作收益承诺。本项目用于学习与实验。

当前未为原创代码指定开源许可证，第三方依赖遵循各自许可证。
