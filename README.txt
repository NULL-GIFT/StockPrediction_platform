1、项目简介
    本项目是基于Flask+ECharts构建的交互式金融量化回测分析平台。
    系统围绕数据清洗→特征工程→多模型训练→策略回测→Web可视化”的设计，
    支持特征因子消融实验、双轨资金回测（含摩擦成本与理想无摩擦）、不同市场行情下的模型抗跌性评估等功能。

2、启动程序
    在项目根目录下执行:python main.py
    在浏览器中打开以下地址即可进入量化回测控制台：http://127.0.0.1:5000
    如需训练有无某一类指标的全部模型指标对比，可在config修改FEATURE_COLUMNS配置参数，运行02_train_models.py即可

3、核心功能模块
    3.1 标的与周期选择：支持选择全市场或单只标的，
                    预测周期支持未来1日/5日
                    打标方式支持横截面相对强弱与绝对涨跌幅阈值进行对照
    3.2 特征消融实验：通过前端复选框动态组合4大类特征
                   基础指标、增强因子、时序记忆、行业板块
                   实现不同特征组下的模型对比实验
    3.3 多模型训练与评估：支持XGBoost、SVM、BP神经网络、逻辑回归四种模型
                      模型训练完成后自动保存，支持缓存加载
                      输出准确率、召回率、F1、AUC、混淆矩阵及ROC曲线
    3.4 策略回测与风控：内置双轨回测引擎（真实摩擦成本、理想无摩擦）
                     支持单票20%动态仓位限制、止盈止损、买入/卖出阈值设置
    3.5 大盘行情交叉体检：基于TimeSeriesSplit进行时序折叠切片
                      根据沪深300区间涨跌幅自动打标为牛市/熊市/震荡
                      评估不同市场环境下各模型的F1表现
    3.6 交互式图表渲染：前端使用ECharts实时渲染
                     K线图
                     多模型评价柱状图
                     混淆矩阵热力图
                     收益对比三线图等

4、文件结构树
StockPrediction_platform
├── app                              # Flask 网页可视化前端模块
│   ├── static
│   │   ├── css
│   │   │   └── style.css
│   │   └── js
│   │       └── dashboard.js
│   ├── templates
│   │   ├── base.html
│   │   └── index.html
│   ├── __init__.py
│   └── routes.py                    # Web接口路由
├── core                             # 项目核心算法模块
│   ├── models                       # 各类预测模型实现
│   │   ├── __init__.py
│   │   ├── base_model.py            # 模型基类
│   │   ├── bp_nn.py                 # BP神经网络
│   │   ├── logistic.py              # 逻辑回归
│   │   ├── svm.py                   # SVM支持向量机
│   │   └── xgboost_clf.py           # XGBoost分类模型
│   ├── __init__.py
│   ├── backtester.py                # 策略回测引擎
│   ├── config.py                    # 全局配置文件
│   ├── evaluator.py                 # 模型通用评估器
│   ├── feature_eng.py               # 特征工程处理
│   └── regime_evaluator.py          # 分市场状态评估器
├── data                             # 全量数据存储目录
│   ├── models                       # 训练好的模型权重文件
│   │   └── Train_20220101_20231231  # 对应时间区间的模型文件夹
│   ├── processed                    # 加工后的特征数据集
│   │   └── features_base.csv
│   └── raw                          # 原始行情数据
│       ├── index300_daily.csv       # 沪深300日线原始数据
│       └── stock_daily.csv          # 个股日线原始数据
├── output                           # 程序所有输出结果
│   ├── figures                      # 图表输出
│   │   └── Train_20220101_20231231
│   ├── regime_evaluator            # 市场状态评估输出
│   │   ├── regime_comparison_label_5d_cs.png
│   │   └── regime_scores_label_5d_cs.csv
│   ├── tables                       # 指标表格输出
│   │   └── Train_20220101_20231231
│   │       ├── Feat_Base_Factor
│   │       └── Feat_Base_Factor_Industry_Lag
│   └── results.json                 # 模型综合结果JSON
├── report                           # 实验报告文档
│   └── 202334071011_魏子涵金融实验报告.pdf
├── scripts                          # 分步执行脚本集合
│   ├── plot_analysis               # 绘图分析子脚本目录
│   ├── 02_train_models.py          # 模型训练脚本
│   ├── fetch_real_data.py           # 拉取真实原始行情数据
│   ├── step01_build_features.py     # 第一步：构建特征数据集
│   └── step03_run_backtest.py      # 第三步：执行策略回测
├── main.py                          # 项目总入口启动文件
├── requirements.txt                 # Python依赖库清单
└── README.txt                       # 项目说明文档


5、各模块功能叙述：
    app：Web可视化平台，基于Flask搭建，提供股票预测结果、回测指标的网页看板展示。
    core：算法核心层，封装特征工程、多机器学习模型、回测逻辑、模型评估与分市场评测工具。
    data：分层存放数据，raw原始行情、processed加工特征、models训练保存的模型文件。
    output：所有实验产出，包含可视化图表、量化指标表格、市场状态评测文件、汇总结果。
    scripts：流水线分步脚本，从数据拉取、特征构建、模型训练到回测完整流程拆分。
    report：实验PDF报告存放目录。
    根目录：main.py一键启动项目，requirements.txt环境配置，README记录使用说明。