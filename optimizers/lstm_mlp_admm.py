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

class LSTM_MLP_ADMM(nn.Module):
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

        # ADMM参数nu生成层
        self.linear_nu = nn.Linear(hidden_size, output_size, bias=use_bias)
        # 松弛参数alpha生成层
        self.linear_alpha = nn.Linear(hidden_size, output_size, bias=use_bias)

        self.state = None
        self.step_size = kwargs.get('step_size', None)

    @property
    def device(self):
        return self.linear_nu.weight.device

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
            模型名称字符串 'LSTM-MLP-ADMM'
        """
        return 'LSTM-MLP-ADMM'

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
            3. MLP生成优化参数(nu,alpha)
            4. 更新X
            5. 更新Z
            6. 更新U
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
        nu = self.linear_nu(output).reshape_as(optimizees.X)
        alpha = self.linear_alpha(output).reshape_as(optimizees.X)
        
        ## 更新原变量 X
        ATA_nuI = torch.bmm(optimizees.W.permute(0, 2, 1), optimizees.W) + nu * torch.eye(optimizees.W.size(2), dtype=optimizees.W.dtype, device=optimizees.W.device).unsqueeze(0).repeat(optimizees.W.size(0), 1, 1)
        rhs = torch.bmm(optimizees.W.permute(0, 2, 1), optimizees.Y) + nu * (optimizees.get_var('Z') - optimizees.get_var('U'))
        X_new = torch.linalg.solve(ATA_nuI, rhs.squeeze(-1)).unsqueeze(-1)
        ## 更新引入变量 Z
        x_relaxed = alpha * X_new + (1 - alpha) * optimizees.get_var('Z')
        Z_temp = x_relaxed + optimizees.get_var('U')
        Z_new = torch.clamp(Z_temp - optimizees.rho / nu, min=0) * torch.sign(Z_temp)
        Z_new = torch.where(Z_temp > optimizees.rho / nu, Z_new, torch.zeros_like(Z_new))
        ## 更新对偶变量 U
        U_new = optimizees.get_var('U') + x_relaxed - Z_new
        ## 更新变量
        optimizees.set_var('X', X_new)
        optimizees.set_var('Z', Z_new)
        optimizees.set_var('U', U_new)

        return optimizees



def test():
    return True


if __name__ == "__main__":
    test()
