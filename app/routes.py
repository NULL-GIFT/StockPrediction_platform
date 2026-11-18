from flask import Blueprint, render_template, request, jsonify
import os
import sys
from core.regime_evaluator import run_offline_regime_eval

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.step03_run_backtest import run_backtest_api

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/api/run_backtest', methods=['POST'])
def handle_backtest():
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Content-Type 必须为 application/json"}), 400

        frontend_data = request.json
        print(f"[Backend Log] 收到前端交互网关数据: {frontend_data}")

        user_params = {}
        try:
            # 文本与路由标识类参数
            user_params['model_name'] = str(frontend_data.get('model_name', 'XGBoost_Enhanced'))
            user_params['train_start'] = str(frontend_data.get('train_start', '2022-01-01'))
            user_params['train_end'] = str(frontend_data.get('train_end', '2023-12-31'))
            user_params['test_start'] = str(frontend_data.get('test_start', '2024-01-01'))
            user_params['test_end'] = str(frontend_data.get('test_end', '2024-12-31'))

            # 拦截前端新增的动态切片与周期控制参数
            user_params['target_code'] = str(frontend_data.get('target_code', 'ALL'))
            user_params['predict_window'] = str(frontend_data.get('predict_window', '5d'))
            # 拦截并接收打标方式
            user_params['label_type'] = str(frontend_data.get('label_type', 'cs'))

            # 接收前端传来的特征消融组合选项
            user_params['selected_features'] = frontend_data.get('selected_features', ['base', 'factor'])

            # 金融数值类参数
            user_params['initial_capital'] = float(frontend_data.get('initial_capital', 100000.0))
            # 底层引擎自动双轨计算
            user_params['buy_threshold'] = float(frontend_data.get('buy_threshold', 0.55))
            user_params['sell_threshold'] = float(frontend_data.get('sell_threshold', 0.45))


            # 风控百分比换算
            tp = frontend_data.get('take_profit')
            user_params['take_profit'] = float(tp) / 100.0 if tp and str(tp).strip() != "" else None

            sl = frontend_data.get('stop_loss')
            user_params['stop_loss'] = float(sl) / 100.0 if sl and str(sl).strip() != "" else None

        except ValueError as e:
            return jsonify({"status": "error", "message": f"参数类型解析异常: {str(e)}"}), 400

        result_data = run_backtest_api(user_params=user_params)

        return jsonify({
            "status": "success",
            "message": "回测执行闭环完毕",
            "data": result_data
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"服务器内部网关崩溃: {str(e)}"}), 500





@main_bp.route('/api/run_regime_eval', methods=['POST'])
def handle_regime_eval():
    try:
        frontend_data = request.json
        print(f"[Backend Log] 收到体检请求, 参数: {frontend_data}")

        # 组装寻址字典
        eval_params = {
            'train_start': str(frontend_data.get('train_start', '2022-01-01')),
            'train_end': str(frontend_data.get('train_end', '2023-12-31')),
            'test_start': str(frontend_data.get('test_start', '2024-01-01')),
            'test_end': str(frontend_data.get('test_end', '2024-12-31')),
            'selected_features': frontend_data.get('selected_features', ['base']),
            'predict_window': str(frontend_data.get('predict_window', '5d')),
            'label_type': str(frontend_data.get('label_type', 'cs'))
        }

        result_data = run_offline_regime_eval(eval_params)

        return jsonify({
            "status": "success",
            "message": "体检完毕并已落盘",
            "data": result_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"体检组件异常: {str(e)}"}), 500