# 南方电网电力负荷预测项目

## 项目说明：
- 基于历史的电力负荷数据，训练XGBOOST模型，实现多变量单步的电力负荷预测
- 本项目旨在开发一个高精度的电力负荷预测模型，为南方电网提供未来短期（如24-168小时）的电力负荷预测能力。
- 通过分析历史负荷数据、气象数据、日期类型等多维度信息，构建机器学习/深度学习模型，实现对电网负荷的准确预测，为电网调度、能源分配和运营决策提供数据支持。

## 项目背景
电力负荷预测是电网运营中的关键环节，准确的预测可以帮助：
- 优化发电计划，降低运营成本
- 提高电网稳定性和供电可靠性
- 支持可再生能源并网决策
- 预防电力短缺或过剩情况

## 数据来源
本项目使用以下数据：
- 历史负荷数据：南方电网提供的2015-2020年小时级负荷数据
- 气象数据：温度、湿度、风速、天气状况等
- 日期信息：节假日、工作日/周末标志
- 经济指标：GDP、工业指数等（可选）

## 项目结构

southern-grid-load-forecasting/
├── data/ # 数据目录
│ ├── raw/ # 原始数据
│ ├── processed/ # 处理后的数据
│ └── external/ # 外部数据（如气象数据）
├── models/ # 训练好的模型
├── notebooks/ # Jupyter笔记本探索性分析
├── src/ # 源代码
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── prediction.py
│ └── evaluation.py
├── config/ # 配置文件
├── results/ # 预测结果和可视化
├── requirements.txt # Python依赖
└── README.md


# 技术栈
- 编程语言: Python 3.8+
- 数据处理: Pandas, NumPy
- 机器学习: Scikit-learn, XGBoost, LightGBM
- 深度学习: TensorFlow, Keras, PyTorch (可选)
- 可视化: Matplotlib, Seaborn, Plotly
- 工作流管理: Prefect/Airflow (可选)


## 安装与配置
1、克隆项目仓库：
git clone https://github.com/your-organization/southern-grid-load-forecasting.git
cd southern-grid-load-forecasting
2、创建虚拟环境并安装依赖：
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
pip install -r requirements.txt
3、配置数据路径和模型参数：
修改config/config.yaml中的路径设置
根据需要调整模型超参数


## 使用方法
### 数据预处理
python src/data_preprocessing.py --input_path data/raw --output_path data/processed

### 特征工程
python src/feature_engineering.py --input_path data/processed --output_path data/features

### 模型训练
python src/model_training.py --model_type xgboost --output_path models/

### 生成预测
python src/prediction.py --model_path models/xgboost_model.pkl --output_path results/

### 评估模型
python src/evaluation.py --predictions_path results/predictions.csv

## 模型性能
### 当前最佳模型表现（在测试集上）：
- MAE（平均绝对误差）：±2.3%
- RMSE（均方根误差）：±3.1%
- MAPE（平均绝对百分比误差）：±2.8%

## 项目进度
- 数据收集与清洗
- 探索性数据分析
- 特征工程
- 基线模型建立
- 高级模型开发（LSTM, XGBoost）
- 模型评估与优化
- 系统集成与部署
- 实时预测管道搭建

## 贡献指南
1、Fork 本仓库
2、创建特性分支 (git checkout -b feature/AmazingFeature)
3、提交更改 (git commit -m 'Add some AmazingFeature')
4、推送到分支 (git push origin feature/AmazingFeature)
5、开启Pull Request


## 许可证
本项目基于MIT许可证 - 查看LICENSE文件了解详情

## 致谢
- 感谢南方电网提供数据支持
- 感谢所有为本项目做出贡献的开发者和研究人员

## 文件说明
- data：数据
- log：日志
- model：保存的模型文件
- src：项目的主要业务逻辑，包括机器学习建模相关的代码
- utils：项目中自定义的工具包

## 更新日志
[1.0.0] - 2025-09-10
- 初始版本发布
- 实现基础特征工程管道
- 建立XGBoost和LSTM基线模型
- 完成初步模型评估