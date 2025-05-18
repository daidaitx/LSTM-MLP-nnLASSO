import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def aggregate_violations(cases, results_dir, output_file):
    """
    整合所有案例的约束违反度数据到单个文件
    
    参数:
        cases: 案例名称列表
        results_dir: 结果目录路径
        output_file: 输出文件名
    """
    violations_data = []
    
    for case in cases:
        case_dir = os.path.join(results_dir, case)
        violations_file = os.path.join(case_dir, 'nnLASSO-violations')
        
        if os.path.exists(violations_file):
            with open(violations_file, 'r') as f:
                violation = float(f.readline().strip())
        else:
            violation = np.nan
            
        violations_data.append({'Case': case, 'Max_Violation': violation})
    
    # 保存为CSV文件
    df = pd.DataFrame(violations_data)
    output_path = os.path.join(results_dir, output_file)
    df.to_csv(output_path, index=False)
    print(f"约束违反度数据已保存至: {output_path}")

def plot_shifted_losses_from_cases(cases, results_dir, fig_name):
    all_losses = []
    max_violations = []
    case_labels = []
    
    # 读取所有案例的损失值和约束违反度
    for case in cases:
        case_dir = os.path.join(results_dir, case)
        losses_file = os.path.join(case_dir, 'nnLASSO-losses')
        violations_file = os.path.join(case_dir, 'nnLASSO-violations')
        
        # 检查文件是否存在
        if not os.path.exists(losses_file):
            print(f"Warning: {losses_file} does not exist. Skipping this case.")
            continue
        
        # 读取损失值
        with open(losses_file, 'r') as file:
            losses = [float(line.strip()) for line in file.readlines()]
            all_losses.append(losses)
            
        # 读取约束违反度
        if os.path.exists(violations_file):
            with open(violations_file, 'r') as file:
                max_violation = float(file.readline().strip())
        else:
            max_violation = np.nan
            
        max_violations.append(max_violation)
        case_labels.append(case.split("-", 1)[1])
    
    plt.figure(figsize=(12, 8))
    
    # 绘制曲线并在图例中显示约束违反度
    for losses, label, violation in zip(all_losses, case_labels, max_violations):
        iterations = range(len(losses))
        if not np.isnan(violation):
            if violation == 0:
                label = f"{label} (0)"
            else:
                label = f"{label} ({violation:.2e})"
        
        abs_losses = [abs(l) for l in losses]
        segments = []
        current_sign = np.sign(losses[0]) if len(losses) > 0 else 1
        start_idx = 0
        
        # 分割正负值区间
        for i in range(1, len(losses)):
            sign = np.sign(losses[i])
            if sign != current_sign:
                segments.append((start_idx, i, current_sign))
                start_idx = i
                current_sign = sign
        segments.append((start_idx, len(losses), current_sign))
        
        # 获取颜色循环中的下一种颜色
        color = plt.gca()._get_lines.get_next_color()
        
        # 绘制各段曲线
        for start, end, sign in segments:
            if start >= end:
                continue
            linestyle = '-' if sign >= 0 else '--'
            plt.plot(iterations[start:end], abs_losses[start:end], 
                    linestyle=linestyle, color=color, 
                    label=label if start == 0 else "")
    
    # 添加标题和标签
    plt.title('Loss Convergence with Constraint Violations')
    plt.xlabel('Iteration')
    plt.ylabel('Relative error of loss, relative to 5000 steps proj-FISTA, log scale')
    
    # 将纵轴改为对数坐标并设置上限
    plt.yscale('log')
    plt.ylim(top=10**2)
    
    # 显示图例
    plt.legend()
    
    # 显示网格
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片到指定目录
    save_path = os.path.join(results_dir, fig_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 指定需要绘制的案例名称
cases_to_plot = ["nnLASSO-noprocessing", "nnLASSO-proj-FISTA",
                 "nnLASSO-x-proj", "nnLASSO-y-proj",
                 "nnLASSO-linear-penalty", "nnLASSO-quadratic-penalty", "nnLASSO-exponential-penalty",
                 "nnLASSO-ADMM"]

# 整合约束违反度数据
aggregate_violations(cases_to_plot, 'results/synthetic-K20', 'nnLASSO-violations-summary.csv')

# 调用函数绘制图形
plot_shifted_losses_from_cases(cases_to_plot, 'results/synthetic-K20', 'synthetic-K20_loss.png')