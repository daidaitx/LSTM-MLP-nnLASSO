import copy
import math
import numpy as np

from optimizees.base import BaseOptimizee

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class proj_FISTA(object):
    def __init__(self, *args, **kwargs):
        """
        proj-FISTA(带投影的快速迭代收缩阈值算法)实现
        用于求解带非光滑正则项的凸优化问题
        
        属性:
            current_optimizees: 当前优化目标
            step_size: 步长
            tau: 动量系数
            Z: 辅助变量(用于动量更新)
        """
        self.current_optimizees = None # 当前优化目标实例
        self.step_size = None          # 梯度下降步长
        self.tau = None                # 动量系数
        self.Z = None                  # 辅助变量(用于Nesterov动量)

    def name(self):
        """
        返回优化器名称
        
        返回:
            str: 优化器名称('proj-FISTA')
        """
        return 'proj-FISTA'

    def detach_state(self):
        """断开与优化目标的引用"""
        pass

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        """
        重置优化器状态
        
        参数:
            optimizees: 优化目标实例
            step_size: 初始步长(如未提供则根据Lipschitz常数计算)
        """
        self.current_optimizees = optimizees  # 设置当前优化目标
        # 计算步长: 默认使用0.9999/L，其中L是梯度Lipschitz常数
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())
        self.tau = 1.0  # 初始化动量系数
        self.Z = copy.deepcopy(optimizees.X)  # 初始化辅助变量

    def __call__(self, optimizees, *args, **kwargs):
        """
        【核心内容】执行一次proj-FISTA优化步骤
        
        参数:
            optimizees: 优化目标实例
        返回:
            更新后的优化目标
            
        算法步骤:
            1. 计算梯度: ∇f(Z_k)
            2. 梯度下降: temp = Z_k - η∇f(Z_k)
            3. 近端+投影: X_{k+1} = max(0, prox_{ηg}(temp))
            4. 更新动量系数: τ_{k+1} = (1+√(1+4τ_k^2))/2
            5. 更新辅助变量: Z_{k+1} = X_{k+1} + ((τ_k-1)/τ_{k+1})(X_{k+1}-X_k)
        """
        step_size = kwargs.get('step_size', self.step_size) # self.step_size为默认步长
        if self.current_optimizees is None or step_size is None:
            raise RuntimeError('优化器未正确初始化，请先调用reset_state方法')
        elif optimizees is not self.current_optimizees:
            print('警告: 检测到新的优化目标，请确保已正确调用reset_state方法')

        # 1. 在辅助变量Z处计算梯度
        smooth_grad = optimizees.smooth_grad(dict(X=self.Z), compute_grad=False)
        # 2. 梯度下降步骤
        temp = self.Z - step_size * smooth_grad
        # 3. 近端算子应用(处理非光滑项) + 投影（使用torch.relu）
        Xnew = torch.relu(optimizees.prox(dict(X=temp, P=step_size)))

        # 4. 更新动量系数(Nesterov加速)
        prev_tau = self.tau
        self.tau = (1.0 + math.sqrt(1.0 + 4.0 * prev_tau**2)) / 2.0
        # 5. 更新辅助变量
        self.Z = Xnew + ((prev_tau - 1.0) / self.tau) * (Xnew - optimizees.X)

        # 更新优化目标状态
        optimizees.X = Xnew

        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()

