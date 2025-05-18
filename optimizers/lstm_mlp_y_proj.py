import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

from collections import defaultdict
from optimizees.base import BaseOptimizee

NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'softplus': nn.Softplus(),
}

class LSTM_MLP_y_Proj(nn.Module):
    """
    LSTM+MLP优化器网络结构
    基于LSTM处理梯度序列，MLP生成优化参数
    实现坐标级非光滑优化算法
    
    主要组件：
    - LSTM: 处理梯度历史信息
    - MLP(p,a,b,b1,b2): 生成预条件、动量等优化参数
    - 近端算子: 处理非光滑项
    
    参考文献:
    Liu et al. (2023) "Towards Constituting Mathematical Structures for Learning to Optimize"
    """
    def __init__(self, input_size, output_size, hidden_size, layers,
                 p_use=True, p_scale=1.0, p_scale_learned=True, p_norm='eye',
                 b_use=True, b_scale=1.0, b_scale_learned=True, b_norm='eye',
                 a_use=True, a_scale=1.0, a_scale_learned=True, a_norm='eye',
                 b1_use=True, b1_scale=1.0, b1_scale_learned=True, b1_norm='eye',
                 b2_use=True, b2_scale=1.0, b2_scale_learned=True, b2_norm='eye',
                 **kwargs):
        """
        初始化LSTM+MLP优化器
        
        参数:
            input_size: 输入维度
            output_size: 输出维度
            hidden_size: LSTM隐藏层维度
            layers: LSTM层数
            p_use: 是否使用预条件矩阵P
            p_scale: P的初始缩放因子
            p_scale_learned: 是否学习P的缩放因子
            p_norm: P的归一化函数(exp/eye/sigmoid/softplus)
            (类似参数a/b/b1/b2用于动量和其他偏置项)
        """
        super().__init__()

        # 网络维度设置
        self.input_size  = input_size   # 输入维度
        self.output_size = output_size  # 输出维度
        self.hidden_size = hidden_size  # LSTM隐藏层维度
        use_bias = True  # 是否使用偏置项

        self.hist = defaultdict(list)  # 训练历史记录

        self.layers = layers  # LSTM层数

        # LSTM核心网络: 处理梯度序列信息
        self.lstm = nn.LSTM(input_size, hidden_size, layers, bias=use_bias)
        
        # 中间MLP层(借鉴NA-ALISTA架构)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        # 预条件矩阵P生成层
        self.linear_p = nn.Linear(hidden_size, output_size, bias=use_bias)
        
        # 混合权重b生成层
        self.linear_b = nn.Linear(hidden_size, output_size, bias=use_bias)
        
        # 偏置项生成层
        self.linear_b1 = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.linear_b2 = nn.Linear(hidden_size, output_size, bias=use_bias)
        
        # 动量系数a生成层
        self.linear_a = nn.Linear(hidden_size, output_size, bias=use_bias)

        self.state = None
        self.step_size = kwargs.get('step_size', None)

        self.p_use = p_use
        if p_scale_learned:
            self.p_scale = nn.Parameter(torch.tensor(1.) * p_scale)
        else:
            self.p_scale = p_scale
        self.p_norm = NORM_FUNC[p_norm]

        self.b_use = b_use
        if b_scale_learned:
            self.b_scale = nn.Parameter(torch.tensor(1.) * b_scale)
        else:
            self.b_scale = b_scale
        self.b_norm = NORM_FUNC[b_norm]

        self.b1_use = b1_use
        if b1_scale_learned:
            self.b1_scale = nn.Parameter(torch.tensor(1.) * b1_scale)
        else:
            self.b1_scale = b1_scale
        self.b1_norm = NORM_FUNC[b1_norm]

        self.b2_use = b2_use
        if b2_scale_learned:
            self.b2_scale = nn.Parameter(torch.tensor(1.) * b2_scale)
        else:
            self.b2_scale = b2_scale
        self.b2_norm = NORM_FUNC[b2_norm]

        self.a_use = a_use
        if a_scale_learned:
            self.a_scale = nn.Parameter(torch.tensor(1.) * a_scale)
        else:
            self.a_scale = a_scale
        self.a_norm = NORM_FUNC[a_norm]

    @property
    def device(self):
        return self.linear_p.weight.device

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        """
        重置LSTM状态
        
        功能:
            初始化或重置LSTM的hidden_state和cell_state
            根据optimizees的batch_size重新初始化状态
            
        参数:
            optimizees: 优化目标实例，用于获取batch_size
            step_size: 初始步长，如果为None则自动计算
            **kwargs: 其他可选参数
            
        实现细节:
            1. 根据optimizees的batch_size初始化状态
            2. 状态包括hidden_state和cell_state
            3. 如果step_size为None，则根据optimizees的梯度Lipschitz常数计算
        """
        batch_size = optimizees.X.numel()
        self.state = (
            # hidden_state
            torch.randn(
                self.layers, batch_size, self.hidden_size
            ).to(self.device),
            # cell_state
            torch.randn(
                self.layers, batch_size, self.hidden_size
            ).to(self.device),
        )
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())

    def detach_state(self):
        """
        分离LSTM状态
        
        功能:
            将LSTM的hidden_state和cell_state从计算图中分离
            用于防止梯度回传
        """
        if self.state is not None:
            self.state = (self.state[0].detach(), self.state[1].detach())

    def name(self):
        """
        获取模型名称
        
        返回:
            模型名称字符串 'LSTM-MLP-y-proj'
        """
        return 'LSTM-MLP-y-proj'

    def forward(
        self,
        optimizees: BaseOptimizee,
        grad_method: str,
        reset_state: bool = False,
        detach_grad: bool = True,
    ):
        """
        【核心内容】执行一步优化计算
        
        参数:
            optimizees: 优化目标实例
            grad_method: 梯度计算方法
            reset_state: 是否重置LSTM状态
            detach_grad: 是否截断梯度计算图
            
        算法步骤:
            1. 获取当前梯度信息
            2. LSTM处理梯度序列
            3. MLP生成优化参数(P,a,b,b1,b2)
            4. 计算参数更新量
            5. 应用近端算子
            6. 更新优化变量
        """
        batch_size = optimizees.batch_size

		## LSTM步骤
        # （可选）初始化或重置LSTM状态
        if self.state is None or reset_state:
            self.reset_state(optimizees)
        # 获取当前梯度信息(用于LSTM输入)
        lstm_input = optimizees.get_grad(
            grad_method=grad_method,
            compute_grad=self.training,
            retain_graph=self.training,
        )
        lstm_input2 = optimizees.X  # 当前优化变量值
        # （可选）截断梯度计算图
        if detach_grad:
            lstm_input = lstm_input.detach()
            lstm_input2 = lstm_input2.detach()
        # 准备LSTM输入数据(合并梯度和当前变量值)
        lstm_input = lstm_input.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_input2 = lstm_input2.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_in = torch.cat((lstm_input,lstm_input2), dim = 2)
        # LSTM核心处理
        output, self.state = self.lstm(lstm_in, self.state)
        output = F.relu(self.linear(output))  # 中间非线性变换
        
		## MLP步骤
        # 生成各优化参数
        P = self.linear_p(output).reshape_as(optimizees.X)  # 预条件矩阵
        B = self.linear_b(output).reshape_as(optimizees.X)   # 混合权重
        A = self.linear_a(output).reshape_as(optimizees.X)  # 动量系数
        B1 = self.linear_b1(output).reshape_as(optimizees.X) # 偏置项1
        B2 = self.linear_b2(output).reshape_as(optimizees.X) # 偏置项2
        # 应用归一化和缩放
        P = self.p_norm(P) * self.p_scale if self.p_use else 1.0
        B = self.b_norm(B) * self.b_scale if self.b_use else 1.0
        A = self.a_norm(A) * self.a_scale if self.a_use else 0.0
        B1 = self.b_norm(B1) * self.b1_scale if self.b1_use else 0.0
        B2 = self.b_norm(B2) * self.b2_scale if self.b2_use else 0.0

		## 近端算子步骤: 更新x
        # 计算梯度更新量
        smooth_grad = optimizees.get_grad(
            grad_method='smooth_grad',
            compute_grad=self.training,
            retain_graph=False
        )
        updateX = - P * self.step_size * smooth_grad  # y的更新量
        smooth_grad2 = optimizees.get_grad(
            grad_method='smooth_grad',
            inputs = {'X':optimizees.get_var('Z')},
            compute_grad=self.training,
            retain_graph=False
        )
        updateZ = - P * self.step_size * smooth_grad2  # x的更新量（PA-only时无用）
        # 应用更新(混合主变量和辅助变量)
        prox_in = B * (optimizees.X + updateX) + (1 - B) * (optimizees.get_var('Z') + updateZ) + B1
        # 应用近端算子(处理非光滑项)
        prox_out = optimizees.prox({'P':P * self.step_size, 'X':prox_in}, compute_grad=self.training)
        
		## 动量步骤: 更新y
        # 计算近端算子输出差异
        prox_diff = prox_out - optimizees.get_var('Z')
        # 更新主变量(加入动量项和偏置) + 投影
        optimizees.X = torch.relu(prox_out + A * prox_diff + B2)
        # 更新辅助变量
        optimizees.set_var('Z', prox_out)

        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()
