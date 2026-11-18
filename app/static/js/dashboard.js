document.addEventListener('DOMContentLoaded', function() {
    var chartKline = echarts.init(document.getElementById('chart-kline'));
    var chartModels = echarts.init(document.getElementById('chart-models'));
    var chartMatrix = echarts.init(document.getElementById('chart-matrix'));
    var chartReturns = echarts.init(document.getElementById('chart-returns'));

    window.addEventListener('resize', function() {
        chartKline.resize();
        chartModels.resize();
        chartMatrix.resize();
        chartReturns.resize();
    });

    document.getElementById('run-btn').addEventListener('click', function() {
        var btn = this;
        var originalText = btn.innerText;
        btn.innerText = "⏳ 引擎运算中，请稍候...";
        btn.disabled = true;
        btn.style.backgroundColor = "#95a5a6";

        var selectedFeatures = ['base']; // 基础指标强制附带
        if(document.getElementById('feat_factor').checked) selectedFeatures.push('factor');
        if(document.getElementById('feat_lag').checked) selectedFeatures.push('lag');
       if(document.getElementById('feat_industry') && document.getElementById('feat_industry').checked) {
            selectedFeatures.push('industry');
        }
        chartKline.showLoading();
        chartModels.showLoading();
        chartMatrix.showLoading();
        chartReturns.showLoading();

        var payload = {
            target_code: document.getElementById('target_code').value,
            predict_window: document.getElementById('predict_window').value,
            // === 核心新增：抓取网页下拉框的值传入后端 ===
            label_type: document.getElementById('label_type').value,
            model_name: document.getElementById('model_name').value,
            train_start: document.getElementById('train_start').value,
            train_end: document.getElementById('train_end').value,
            test_start: document.getElementById('test_start').value,
            test_end: document.getElementById('test_end').value,
            selected_features: selectedFeatures,
            initial_capital: parseFloat(document.getElementById('initial_capital').value),
            take_profit: document.getElementById('take_profit').value,
            stop_loss: document.getElementById('stop_loss').value,
            buy_threshold: parseFloat(document.getElementById('buy_threshold').value),
            sell_threshold: parseFloat(document.getElementById('sell_threshold').value)
        };

        fetch('/api/run_backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            if(data.status === 'success') {
                var result = data.data;

                document.getElementById('metric-return').innerText = result.metrics.total_return;
                document.getElementById('metric-alpha').innerText = result.metrics.alpha;
                document.getElementById('metric-drawdown').innerText = result.metrics.max_drawdown;
                document.getElementById('metric-sharpe').innerText = result.metrics.sharpe_ratio;

                // 学术数据填充
                if (result.academic_metrics) {
                    document.getElementById('metric-acc').innerText = result.academic_metrics.accuracy;
                    document.getElementById('metric-prec').innerText = result.academic_metrics.precision;
                    document.getElementById('metric-rec').innerText = result.academic_metrics.recall;
                    document.getElementById('metric-f1').innerText = result.academic_metrics.f1_score;
                    document.getElementById('metric-auc').innerText = result.academic_metrics.auc;
                }

                chartKline.hideLoading();
                chartModels.hideLoading();
                chartMatrix.hideLoading();
                chartReturns.hideLoading();

                // 数据绑定重绘
                renderRealKline(chartKline, result.kline_data);
                renderRealModels(chartModels, result.comparison_data);
                renderRealMatrix(chartMatrix, result.matrix_data);
                renderReturnsChart(chartReturns, result.chart_data);
            } else {
                alert("引擎报错: " + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert("网关响应异常，请检查控制台");
        })
        .finally(() => {
            btn.innerText = originalText;
            btn.disabled = false;
            btn.style.backgroundColor = "#27ae60";
        });
    });




    var chartRegime = null; // 延迟初始化

    document.getElementById('close-modal-btn').addEventListener('click', function() {
        document.getElementById('regime-modal').style.display = 'none';
    });

    document.getElementById('eval-regime-btn').addEventListener('click', function() {
        var btn = this;
        var originalText = btn.innerText;
        btn.innerText = "⏳ 正在切片与离线评估...";
        btn.disabled = true;

        // 复用前台的参数，告诉后端去哪里找模型
        var selectedFeatures = ['base'];
        if(document.getElementById('feat_factor').checked) selectedFeatures.push('factor');
        if(document.getElementById('feat_lag').checked) selectedFeatures.push('lag');
        if(document.getElementById('feat_industry') && document.getElementById('feat_industry').checked) {
            selectedFeatures.push('industry');
        }
        var payload = {
            train_start: document.getElementById('train_start').value,
            train_end: document.getElementById('train_end').value,
            // === 架构补全：接通测试集前端输入管道 ===
            test_start: document.getElementById('test_start').value,
            test_end: document.getElementById('test_end').value,
            selected_features: selectedFeatures,
            predict_window: document.getElementById('predict_window').value,
            label_type: document.getElementById('label_type').value
        };

        fetch('/api/run_regime_eval', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            if(data.status === 'success') {
                var result = data.data; // 提前拿到 data

                // 【新增】：动态修改弹窗的大标题
                var titleEl = document.getElementById('modal-title');
                if(titleEl) {
                    titleEl.innerText = "📊 策略抗跌性评估 | " + result.eval_target;
                }

                // 显示弹窗
                document.getElementById('regime-modal').style.display = 'flex';
                // 初始化弹窗内的ECharts
                if (!chartRegime) {
                    chartRegime = echarts.init(document.getElementById('chart-regime'));
                }
                // 画图
                var result = data.data;
                chartRegime.setOption({
                    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                    legend: { data: result.models, top: 0 },
                    grid: { left: '8%', right: '4%', bottom: '3%', containLabel: true },
                    xAxis: { type: 'category', data: result.regime_tags },
                    yAxis: {
                        type: 'value',
                        // 动态计算 Y 轴下限：从数据最低点再往下减 0.05，让柱子全冒出来
                        min: function(value) {
                            return Math.max(0, (value.min - 0.05).toFixed(2));
                        },
                        name: '综合指标 (F1)'
                    },
                    series: result.series_data
                });
            } else {
                alert("体检失败: " + data.message);
            }
        })
        .catch(error => { console.error('Error:', error); alert("网关响应异常"); })
        .finally(() => {
            btn.innerText = originalText;
            btn.disabled = false;
        });
    });


});



function renderReturnsChart(chartInstance, chartData) {

    var baseValue = chartData.strategy_assets[0] || 100000;

    var option = {
        // 悬浮提示框双显拦截
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' },
            formatter: function (params) {
                var html = params[0].name + '<br/>'; // 顶部显示日期
                params.forEach(function (item) {
                    var val = item.value;

                    var pct = ((val - baseValue) / baseValue * 100).toFixed(2);

                    html += item.marker + item.seriesName + ': ' + pct + '% (' + val.toFixed(0) + '元)<br/>';
                });
                return html;
            }
        },
        legend: { data: ['理想策略净值', '真实策略净值(扣除成本)', '买入并持有基准', '沪深300基准'], top: 0 },
        grid: { left: '4%', right: '5%', bottom: '5%', containLabel: true },
        dataZoom: [{ type: 'inside', start: 0, end: 100 }, { start: 0, end: 100 }],

        xAxis: { type: 'category', boundaryGap: false, data: chartData.dates },


        yAxis: {
            type: 'value',
            scale: true,
            name: '累计收益率 (%)', // 修改名字以符合任务书要求
            splitLine: { lineStyle: { type: 'dashed' } },
            axisLabel: {
                formatter: function (value) {
                    var pct = ((value - baseValue) / baseValue * 100).toFixed(2);
                    return pct + '%';
                }
            }
        },

        series: [
            // 无摩擦理想收益曲线（紫线）
            {
                name: '理想策略净值',
                type: 'line',
                data: chartData.strategy_assets_ideal,
                lineStyle: { color: '#9b59b6', width: 2, type: 'dashed' },
                showSymbol: false,
                label: {
                    show: true,
                    position: 'top',
                    formatter: function(params) {
                        if (params.dataIndex === chartData.strategy_assets_ideal.length - 1) {
                            var pct = ((params.value - baseValue) / baseValue * 100).toFixed(2);
                            return '理想: ' + pct + '% (' + params.value.toFixed(0) + ')';
                        } else {
                            return '';
                        }
                    }
                }
            },
            // 真实策略净值（红线）
            {
                name: '真实策略净值(扣除成本)',
                type: 'line',
                data: chartData.strategy_assets,
                lineStyle: { color: '#e74c3c', width: 2 },
                showSymbol: false,
                label: {
                    show: true,
                    position: 'bottom',
                    formatter: function(params) {
                        if (params.dataIndex === chartData.strategy_assets.length - 1) {
                            var pct = ((params.value - baseValue) / baseValue * 100).toFixed(2);
                            return '真实: ' + pct + '% (' + params.value.toFixed(0) + ')';
                        } else {
                            return '';
                        }
                    }
                }
            },
            // 基准线
            { name: '买入并持有基准', type: 'line', data: chartData.buy_and_hold_assets, lineStyle: { color: '#f39c12', width: 2 }, showSymbol: false },
            { name: '沪深300基准', type: 'line', data: chartData.benchmark_assets, lineStyle: { color: '#3498db', width: 2 }, showSymbol: false }
        ]
    };
    chartInstance.setOption(option, true);
}



// K线
function renderRealKline(chartInstance, klineData) {
    var option = {
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        // 增加图例显示
        legend: { data: ['个股日K', 'MA5均线', 'MA10均线', 'MA20均线'], top: 0 },
        grid: { left: '5%', right: '5%', bottom: '15%' },
        dataZoom: [{ type: 'inside', start: 70, end: 100 }, { start: 70, end: 100 }],
        xAxis: { type: 'category', data: klineData.dates, scale: true, boundaryGap: true },
        yAxis: { scale: true, splitLine: { lineStyle: { type: 'dashed' } } },
        series: [
            {
                name: '个股日K',
                type: 'candlestick',
                data: klineData.values,
                itemStyle: {
                    color: '#e74c3c',
                    color0: '#2cc7c9',
                    borderColor: '#e74c3c',
                    borderColor0: '#2cc7c9'
                }
            },
            { name: 'MA5均线', type: 'line', data: klineData.ma5, smooth: true, lineStyle: { opacity: 0.7, color: '#9b59b6', width: 1.5 }, showSymbol: false },
            { name: 'MA10均线', type: 'line', data: klineData.ma10, smooth: true, lineStyle: { opacity: 0.7, color: '#f1c40f', width: 1.5 }, showSymbol: false },
            { name: 'MA20均线', type: 'line', data: klineData.ma20, smooth: true, lineStyle: { opacity: 0.7, color: '#3498db', width: 1.5 }, showSymbol: false }
        ]
    };
    chartInstance.setOption(option, true);
}






function renderRealModels(chartInstance, compData) {
    var option = {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        legend: { top: 0 },
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
        xAxis: { type: 'category', data: compData.models },
        yAxis: { type: 'value', min: 0.3, max: 0.8, name: '指标得分' },
        series: [
            { name: 'Accuracy (准确率)', type: 'bar', data: compData.accuracy, itemStyle: { color: '#3498db' } },
            { name: 'AUC (区分度)', type: 'bar', data: compData.auc, itemStyle: { color: '#2ecc71' } }
        ]
    };
    chartInstance.setOption(option, true);
}

function renderRealMatrix(chartInstance, matrixData) {
    var option = {
        tooltip: { position: 'top' },
        grid: { left: '15%', right: '5%', bottom: '20%', top: '10%' },
        xAxis: { type: 'category', data: ['预测看跌(0)', '预测看涨(1)'], splitArea: { show: true } },
        yAxis: { type: 'category', data: ['真实下跌(0)', '真实上涨(1)'], splitArea: { show: true } },
        visualMap: {
            min: 0, max: 2500, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%',
            inRange: { color: ['#f9f9f9', '#ff9900', '#dd2200'] }
        },
        series: [{
            name: '样本计数', type: 'heatmap', data: matrixData,
            label: { show: true, formatter: function(p) { return p.data[2]; }, fontStyle: 'bold' },
            emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
        }]
    };
    chartInstance.setOption(option, true);
}