import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from .base import BaseOptimizee


class nnLASSO(BaseOptimizee):

	# 初始化：生成/加载nnLASSO优化问题
    def __init__(
        self,
        batch_size: int,
        W = None,
        Y = None,
        rho = 0.1,
        s = 5,
        device = 'cpu',
        **options
    ) -> None:
        """初始化nnLASSO优化问题。
        
        self参数说明:
            self是Python类方法的第一个参数，代表类的当前实例。
            通过self可以访问实例的属性和方法，如self.device访问设备属性。
            所有实例方法都必须将self作为第一个参数，但调用时不需要显式传递。
            
        参数:
            batch_size (int): 批量大小
            W (可选): 权重矩阵，默认为随机生成
            Y (可选): 观测值，默认为随机生成
            rho (float): L1正则化系数，默认为0.1
            s (int): 稀疏度(非零元素数量)，默认为5
            device (str): 计算设备，默认为'cpu'
            **options: 其他选项，包括:
                input_dim (int): 输入维度
                output_dim (int): 输出维度
                seed (int): 随机种子
                
        优化问题形式:
            minimize (1/2) * ||Y - W @ X||_2^2  + rho * ||X||_1
               X≥0
        """
        self.device = device  # 设置计算设备(cpu/gpu)
        self.vars = dict()    # 初始化变量字典，用于存储优化过程中的变量(X,Z等)

        # 设置基本参数
        self.batch_size = batch_size  # 批量大小
        self.input_dim  = options.get('input_dim')   # 输入维度
        self.output_dim = options.get('output_dim')  # 输出维度
        self.rho = rho        # L1正则化系数
        self.s = s            # 稀疏度(非零元素数量)

        # 处理随机种子
        seed = options.get('seed', None)
        if seed:
            rng_state = torch.get_rng_state()  # 保存当前随机状态
            torch.manual_seed(seed)            # 设置随机种子

        # 初始化权重矩阵W
        if W is None:
            # 随机生成W并归一化
            W = torch.randn(self.batch_size, self.output_dim, self.input_dim).to(self.device)
            self.W = W / torch.sum(W**2, dim=1, keepdim=True).sqrt()
        else:
            # 处理用户提供的W
            if isinstance(W, np.ndarray):
                W = torch.from_numpy(W).to(self.device)
            elif isinstance(W, torch.Tensor):
                W = W.to(self.device)
            else:
                raise ValueError(f'Invalid type {type(W)} for W')
            assert W.dim() == 2  # 确保W是2维张量
            self.W = W.unsqueeze(0).repeat(self.batch_size, 1, 1)  # 扩展为batch形式

        # 初始化观测值Y
        if Y is None:
            # 随机生成非负X_gt(真实值)和对应的Y
            X_gt = torch.abs(torch.randn(batch_size, self.input_dim).to(self.device))
            # 随机选择s个非零位置
            non_zero_idx = torch.multinomial(
                torch.ones_like(X_gt), num_samples=self.s, replacement=False
            )
            # 创建稀疏X_gt
            self.X_gt = torch.zeros_like(X_gt).scatter(
                dim=1, index=non_zero_idx, src=X_gt
            ).unsqueeze(-1)
            self.Y = torch.bmm(self.W, self.X_gt)  # 计算Y=WX
        else:
            # 处理用户提供的Y
            if isinstance(Y, np.ndarray):
                Y = torch.from_numpy(Y).to(self.device)
            elif isinstance(Y, torch.Tensor):
                Y = Y.to(self.device)
            else:
                raise ValueError(f'Invalid type {type(Y)} for Y')
            assert Y.dim() == 2
            self.Y = Y.unsqueeze(0).repeat(self.batch_size, 1, 1)  # 扩展为batch形式
            self.X_gt = None  # 无真实值X_gt

        # 初始化优化变量X
        self.initialize()  # X在论文中对应Y, prox_out对应X

        # 恢复原始随机状态(如果设置了种子)
        if seed:
            rng_state = torch.set_rng_state(rng_state)

    def initialize(self):
        """初始化优化变量。
        
        初始化:
        - X: 形状为(batch_size, input_dim, 1)的零张量
        - Z: 形状为(batch_size, input_dim, 1)的零张量，存储在vars字典中
        """
        self.X = torch.zeros(self.batch_size, self.input_dim, 1).to(self.device)
        prox_out = torch.zeros(self.batch_size, self.input_dim, 1).to(self.device)
        self.set_var('Z', prox_out)
        self.set_var('U', prox_out)

	# 变量字典self.vars的获取、设置、分离
	# 		- X（动量步后的y）
	# 		- Z（近端算子后的x）
    # 		- U（对偶变量，仅ADMM中使用）
    def get_var(self, var_name):
        """从内部变量字典中获取变量。
        
        参数:
            var_name (str): 要获取的变量名称
            
        返回:
            torch.Tensor: 请求的变量值
        """
        return self.vars[var_name]

    def set_var(self, var_name, var_value):
        """在内部变量字典中设置变量。
        
        参数:
            var_name (str): 要设置的变量名称
            var_value (torch.Tensor): 变量的值
        """
        self.vars[var_name] = var_value

    def detach_vars(self):
        """将所有变量从计算图中分离。
        
        分离vars字典中存储的所有张量，防止梯度计算。
        """
        # for var in self.vars.values():
        #     var.detach_()
            
        for var_name in self.vars:
            var = self.get_var(var_name)
            var_detached = var.detach()
            self.set_var(var_name, var_detached)


	# 获取、设置优化变量X（动量步后的y）
    @property
    def X(self):
        """获取当前优化变量X。
        
        返回:
            torch.Tensor: X变量的当前值
        """
        return self.get_var('X')

    @X.setter
    def X(self, value):
        """设置当前优化变量X。
        
        参数:
            value (torch.Tensor): X变量的新值
        """
        self.set_var('X', value)

	# 计算Lipschitz常数
    def grad_lipschitz(self):
        """计算平滑部分梯度的Lipschitz常数。
        
        返回:
            torch.Tensor: 每个样本的Lipschitz常数，
                         形状为(batch_size, 1, 1)
        """
        lip = torch.linalg.norm(self.W, dim=(-2,-1), ord=2) ** 2
        return lip.reshape(self.batch_size, 1, 1)

	# 计算目标函数值（带偏移和不带偏移、批处理版本和非批处理版本）
    def objective(self, inputs: dict = None, compute_grad: bool = False):
        """计算目标函数值。
        
        参数:
            inputs (dict, 可选): 包含要使用变量的字典。
                如果未提供，则使用类变量。
            compute_grad (bool, 可选): 是否计算梯度。
            
        返回:
            torch.Tensor: 目标函数值（所有样本的均值）
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            residual = torch.bmm(self.W, X) - self.Y
            l2 = 0.5 * (residual**2.0).sum(dim=(1,2)).mean()
            l1 = self.rho * torch.abs(X).sum(dim=(1,2)).mean()
            return l1 + l2

    def objective_batch(self, inputs: dict = {}, compute_grad: bool = False):
        """计算批处理的目标函数值。
        
        参数:
            inputs (dict, 可选): 包含要使用变量的字典。
                如果未提供，则使用类变量。
            compute_grad (bool, 可选): 是否计算梯度。
            
        返回:
            torch.Tensor: 每个样本的目标函数值
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            residual = torch.bmm(self.W, X) - self.Y
            l2 = 0.5 * (residual**2.0).sum(dim=(1,2))
            l1 = self.rho * torch.abs(X).sum(dim=(1,2))
            return l1 + l2

    def objective_batch_shift(self, inputs: dict = {}, compute_grad: bool = False):
        """计算批处理的偏移目标函数值。
        
        偏移目标函数值为(f(x) - f*)/f*，其中f*为最优值。
        
        参数:
            inputs (dict, 可选): 包含要使用变量的字典。
                如果未提供，则使用类变量。
            compute_grad (bool, 可选): 是否计算梯度。
            
        返回:
            torch.Tensor: 每个样本的偏移目标函数值
        """
        if inputs is None:
            inputs = {}
        obj = self.objective_batch(inputs, compute_grad)
        fstar = self.fstar.reshape_as(obj)
        valid_ind = (fstar != 0)
        ret = obj - fstar 
        ret[valid_ind] /= fstar[valid_ind]
        return ret
        
    def objective_shift(self, inputs: dict = {}, compute_grad: bool = False):
        """计算批处理的平均偏移目标函数值。
        
        参数:
            inputs (dict, 可选): 包含要使用变量的字典。
                如果未提供，则使用类变量。
            compute_grad (bool, 可选): 是否计算梯度。
            
        返回:
            torch.Tensor: 批处理的平均偏移目标函数值
        """
        if inputs is None:
            inputs = {}
        return self.objective_batch_shift(inputs, compute_grad).mean()

	# 计算目标函数的梯度（包括PyTorch内置的反向传播梯度、光滑项梯度、次梯度）
    def get_grad(
        self,
        grad_method: str,
        inputs: dict = None,
        compute_grad: bool = False,
        **kwargs
    ):
        """计算指定类型的梯度。
        
        参数:
            grad_method (str): 梯度计算方法，可选['smooth_grad', 'subgrad', 'bp_grad']
            inputs (dict, 可选): 包含要使用变量的字典。
                如果未提供，则使用类变量。
            compute_grad (bool, 可选): 是否计算梯度。
            **kwargs: 其他关键字参数
            
        返回:
            torch.Tensor: 计算得到的梯度
        """
        grad_func = getattr(self, grad_method, None)
        assert grad_func, f'Invalid grad method specified: {grad_method}'
        return grad_func(inputs, compute_grad, **kwargs)

    def bp_grad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """使用PyTorch内置的反向传播（bp: backpropagation）机制计算目标函数对迭代变量的梯度。
        
        参数:
            inputs (dict, 可选): 包含要使用变量的字典。
                如果未提供，则使用类变量。
            compute_grad (bool, 可选): 是否计算梯度。
            **kwargs: 其他关键字参数
            
        返回:
            torch.Tensor: 计算得到的梯度
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        X = X.detach()
        X.requires_grad_(True)
        X.retain_grad()
        inputs['X'] = X

        # Calculate the objective value
        objective = self.objective(inputs=inputs, compute_grad=True)

        # Run backpropagation with computation graph retained, if necessary
        objective.backward(retain_graph=compute_grad)

        return X.grad

    def smooth_grad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """计算目标函数光滑项梯度。
        
        参数:
            inputs (dict, 可选): 包含要使用变量的字典。
                如果未提供，则使用类变量。
            compute_grad (bool, 可选): 是否计算梯度。
            **kwargs: 其他关键字参数
            
        返回:
            torch.Tensor: 计算得到的梯度
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X) # self.X 为默认值
        Y = inputs.get('Y', self.Y) # self.Y 为默认值
        W = inputs.get('W', self.W) # self.W 为默认值

        with torch.set_grad_enabled(compute_grad):
            residual = torch.bmm(W, X) - Y
            return torch.bmm(W.permute(0,2,1), residual) # 返回 W'(WX-Y)

    def subgrad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """计算目标函数的次梯度。
        
        参数:
            inputs (dict, 可选): 包含要使用变量的字典。
                如果未提供，则使用类变量。
            compute_grad (bool, 可选): 是否计算梯度。
            **kwargs: 其他关键字参数
            
        返回:
            torch.Tensor: 计算得到的次梯度
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            residual = torch.bmm(W, X) - Y
            return (torch.bmm(W.permute(0,2,1), residual) +
                    self.rho * torch.sign(X))

	# 计算近端算子prox_out（近端算子后的x）
    def prox(self, inputs: dict, compute_grad: bool = False, **kwargs):
        """计算近端算子。
        
        参数:
            inputs (dict): 必须包含'P'和'X'键
            compute_grad (bool, 可选): 是否计算梯度。
            **kwargs: 其他关键字参数
            
        返回:
            torch.Tensor: 近端算子结果
        """
        P = inputs['P']
        X = inputs['X']
        with torch.set_grad_enabled(compute_grad):
            mag = nn.functional.relu(torch.abs(X) - self.rho * P)
            return torch.sign(X) * mag

	# 文件、解的保存和加载
    def save_to_file(self, path):
        """将当前优化问题的参数保存到.mat文件。
        
        参数:
            path (str): 保存路径
            
        保存内容:
            W: 权重矩阵
            Y: 观测值
            rho: 正则化系数
        """
        Wcpu = self.W.cpu().numpy()
        Ycpu = self.Y.cpu().numpy()
        sio.savemat(path, {'W':Wcpu, 'Y':Ycpu, 'rho':self.rho})

    def load_from_file(self, path):
        """从.mat文件加载优化问题参数。
        
        参数:
            path (str): 加载路径
            
        加载内容:
            W: 权重矩阵
            Y: 观测值
            rho: 正则化系数
        """
        mats = sio.loadmat(path)
        self.W = torch.from_numpy(mats['W']).type(torch.float32).to(self.device)
        self.Y = torch.from_numpy(mats['Y']).type(torch.float32).to(self.device)
        rho = mats['rho']
        while isinstance(rho, np.ndarray):
            rho = rho[0]
        self.rho = rho
        self.batch_size = self.W.shape[0]
        self.output_dim = self.W.shape[1]
        self.input_dim = self.W.shape[2]
        self.Y = torch.reshape(self.Y, (self.batch_size, self.output_dim, 1))

    def save_sol(self, sol, path):
        """将最优解保存到.mat文件。
        
        参数:
            sol: 最优解
            path (str): 保存路径
        """
        sio.savemat(path, {'fstar':sol})

    def load_sol(self, path):
        """从.mat文件加载最优解。
        
        参数:
            path (str): 加载路径
        """
        fstar = sio.loadmat(path)['fstar']
        self.fstar = torch.from_numpy(fstar).to(self.device)

