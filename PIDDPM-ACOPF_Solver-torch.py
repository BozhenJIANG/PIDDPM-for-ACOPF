import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import math
import time

from case118 import *  # 118节点系统
from pypower.api import runpf, ppoption
torch.cuda.set_device(1)  # 强制使用GPU 1
device = torch.device('cuda:1')
print(f"使用设备: {device}")
import gc

NUM_GENERATORS = 54  # 118系统有54台发电机
NUM_BUSES = 118      # 118个母线

case118 = case118()

def RMSE(data1, data2):
    data1 = data1.reshape(-1)
    data2 = data2.reshape(-1)
    return np.sqrt(np.mean((data1 - data2) ** 2))

def MAE(data1, data2):
    data1 = data1.reshape(-1)
    data2 = data2.reshape(-1)
    return np.mean(np.abs((data1 - data2)))

class BetaSiLU(nn.Module):
    def __init__(self, beta=1.0):
        super(BetaSiLU, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class MinMaxSigmoid(nn.Module):
    def forward(self, x):
        return 0.9399 + 0.1201 * torch.sigmoid(x)

def calculate_ybus(branch_data, num_buses, bus_data):
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    for _i, branch in enumerate(branch_data):
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])
          
        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2)) / ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        
        Ybus[from_bus, to_bus] -= Y * (ratio * np.exp(1j * angle_rad)) / ratio**2
        Ybus[to_bus, from_bus] -= Y * (ratio * np.exp(-1j * angle_rad)) / ratio**2

    for i in range(num_buses):
        Gs = bus_data[i][4]
        Bs = bus_data[i][5]
        Ybus[i, i] += Gs / 100 + 1j * Bs / 100
    
    return torch.tensor(Ybus, dtype=torch.complex64).to(device)

def calculate_ybus_(branch_data, num_buses, bus_data):
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    for _i, branch in enumerate(branch_data):
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])
          
        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2)) / ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        
        Ybus[from_bus, to_bus] -= Y * (ratio * np.exp(1j * angle_rad)) / ratio**2
        Ybus[to_bus, from_bus] -= Y * (ratio * np.exp(-1j * angle_rad)) / ratio**2

    return torch.tensor(Ybus, dtype=torch.complex64).to(device)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        return x + self.net(x)

class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)
    
    def forward(self, t):
        t = t.float()
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class OptimizedDDPM(nn.Module):
    def __init__(self, act_dim, state_dim, hidden_dim=512, num_layers=6, time_dim=128):
        super().__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 动作分支
        self.action_branch = self._build_branch(act_dim, hidden_dim, num_layers)
        
        # 状态分支  
        self.state_branch = self._build_branch(state_dim, hidden_dim, num_layers)
        
        # 合并网络
        self.merge_net = self._build_merge_network(hidden_dim * 2 + time_dim, hidden_dim, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, act_dim)
        
    def _build_branch(self, input_dim, hidden_dim, num_layers):
        """构建分支网络"""
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim, hidden_dim * 2))
            
        return nn.Sequential(*layers)
    
    def _build_merge_network(self, input_dim, hidden_dim, num_layers):
        """构建合并网络"""
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim, hidden_dim * 2))
            
        return nn.Sequential(*layers)
    
    def forward(self, y, t, x):
        """
        前向传播
        y: 噪声动作 [batch_size, act_dim]
        t: 时间步 [batch_size, 1]  
        x: 状态 [batch_size, state_dim]
        """
        # 时间嵌入
        t_emb = self.time_mlp(t.squeeze(-1))  # [batch_size, time_dim]
        
        # 分支处理
        h_action = self.action_branch(y)  # [batch_size, hidden_dim]
        h_state = self.state_branch(x)    # [batch_size, hidden_dim]
        
        # 合并特征
        merged = torch.cat([h_action, h_state, t_emb], dim=-1)  # [batch_size, hidden_dim*2 + time_dim]
        
        # 合并网络处理
        merged_out = self.merge_net(merged)
        
        # 输出
        output = self.output_layer(merged_out)
        
        return output

class PINN_PF_Model(nn.Module):
    def __init__(self, act_dim, state_dim, intermediate_dim, limits_q):
        super(PINN_PF_Model, self).__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.intermediate_dim = intermediate_dim
        self.limits_q = limits_q
        
        # Action branch
        self.x_branch = nn.Sequential(
            nn.Linear(act_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, 4 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(4 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(8 * intermediate_dim, 16 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(16 * intermediate_dim, 32 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(32 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU()
        )
        
        # State branch
        self.y_branch = nn.Sequential(
            nn.Linear(state_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, 4 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(4 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(8 * intermediate_dim, 16 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(16 * intermediate_dim, 32 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(32 * intermediate_dim, 8 * intermediate_dim),
            nn.SiLU()
        )
        
        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(16 * intermediate_dim, 16 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(16 * intermediate_dim, 32 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(32 * intermediate_dim, 64 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(64 * intermediate_dim, 32 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(32 * intermediate_dim, 236)
        )
        
        self.beta_silu = BetaSiLU(beta=1.0)
        self.min_max_sigmoid = MinMaxSigmoid()

    def forward(self, x, y):
        x_feat = self.x_branch(x)
        y_feat = self.y_branch(y)
        
        combined = torch.cat([x_feat, y_feat], dim=1)
        p7 = self.combined_net(combined)
        
        # Process outputs
        p7_0 = torch.sigmoid(p7[:, :54])
        p8_1 = torch.stack([a + (b - a) * p7_0[:, i] for i, (a, b) in enumerate(self.limits_q)], dim=1)
        
        p8_2 = self.min_max_sigmoid(p7[:, 54:118])
        p8_3 = self.beta_silu(p7[:, 118:])
        
        p8 = torch.cat([p8_1, p8_2, p8_3], dim=1)
        return p8

class DPPM_class(nn.Module):
    def __init__(self, act_dim, intermediate_dim, state_dim, latent_dim, model_type="FC", l1_reg=0.01):
        super(DPPM_class, self).__init__()
        self.act_dim = act_dim
        self.intermediate_dim = intermediate_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.model_type = model_type
        self.l1_reg = l1_reg
        self.limits_q = [(_temp_data[4]/100, _temp_data[3]/100) for _temp_data in case118["gen"]]
                
        self.DDPM = self.ddpm_model()
        self.PINN_PF = self.pinn_pf_model()

    def pinn_pf_model(self):
        return PINN_PF_Model(self.act_dim, self.state_dim, self.intermediate_dim, self.limits_q)

    def pinn_pf(self, x, y):
        return self.PINN_PF(x, y)

    def ddpm_model(self):
        if self.model_type == "FC":
            return OptimizedDDPM(self.act_dim, self.state_dim)
    
    def ddpm(self, y):
        return self.DDPM(y)

# 数据加载
print("加载数据...")
X_con_train_ = []
with open("./Dataset/X_con_118_train.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_con_train_.append([float(_i) for _i in _data])

X_in_train_ = []
with open("./Dataset/X_in_118_train.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_in_train_.append([float(_i) for _i in _data])

X_other_information_train_ = []
with open("./Dataset/X_other_information_118_train.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_other_information_train_.append([float(_i) for _i in _data])

X_con_test_ = []
with open("./Dataset/X_con_118_test.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_con_test_.append([float(_i) for _i in _data])

X_in_test_ = []
with open("./Dataset/X_in_118_test.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_in_test_.append([float(_i) for _i in _data])

X_other_information_test_ = []
with open("./Dataset/X_other_information_118_test.txt", "r") as f:
    for line in f.readlines():
        _data = line.split()
        X_other_information_test_.append([float(_i) for _i in _data])


# 数据预处理
scaler1 = MinMaxScaler(feature_range=(-1, 1))
scaler2 = MinMaxScaler(feature_range=(-1, 1))
scaler3 = MinMaxScaler(feature_range=(-1, 1))

# 测试标志
test_flag = 0

if test_flag == 0:
    X_in_train = np.array(X_in_train_)
    X_in_train[:, :NUM_GENERATORS] = X_in_train[:, :NUM_GENERATORS] / 100  # 发电机功率归一化
    X_con_train = np.array(X_con_train_) / 100
    X_other_information_train = np.array(X_other_information_train_)

    X_in_test = np.array(X_in_test_)
    X_in_test[:, :NUM_GENERATORS] = X_in_test[:, :NUM_GENERATORS] / 100
    X_con_test = np.array(X_con_test_) / 100
    X_other_information_test = np.array(X_other_information_test_)
else:
    # 测试模式，使用少量数据
    X_in_train = np.array(X_in_train_)[:1000, :]
    X_in_train[:, :NUM_GENERATORS] = X_in_train[:, :NUM_GENERATORS] / 100
    X_con_train = np.array(X_con_train_)[:1000, :] / 100
    X_other_information_train = np.array(X_other_information_train_)[:1000, :]

    X_in_test = np.array(X_in_train_)[:1000, :]
    X_in_test[:, :NUM_GENERATORS] = X_in_test[:, :NUM_GENERATORS] / 100
    X_con_test = np.array(X_con_train_)[:1000, :] / 100
    X_other_information_test = np.array(X_other_information_train_)[:1000, :]

scaler_flag = 0 # 1进行归一化 0不进行归一化

if scaler_flag == 1:
    # 拟合scaler
    scaler1.fit(np.vstack((X_in_train, X_in_test)))
    scaler2.fit(np.vstack((X_con_train, X_con_test)))
    scaler3.fit(np.vstack((X_other_information_train, X_other_information_test)))

    # 转换数据
    X_in_train = scaler1.transform(X_in_train)
    X_con_train = scaler2.transform(X_con_train)
    X_other_information_train = scaler3.transform(X_other_information_train)

    X_in_test = scaler1.transform(X_in_test)
    X_con_test = scaler2.transform(X_con_test)
    X_other_information_test = scaler3.transform(X_other_information_test)

# 转换为PyTorch张量
X_in_train_tensor = torch.FloatTensor(X_in_train).to(device)
X_con_train_tensor = torch.FloatTensor(X_con_train).to(device)
X_other_information_train_tensor  = torch.tensor(X_other_information_train, dtype=torch.float32)

X_in_test_tensor = torch.FloatTensor(X_in_test).to(device)
X_con_test_tensor = torch.FloatTensor(X_con_test).to(device)
X_other_information_test_tensor  = torch.tensor(X_other_information_test, dtype=torch.float32)

print(X_other_information_train[:, :54].min()," ",X_other_information_train[:, :54].max())
print(X_other_information_train[:, 54:118].min()," ",X_other_information_train[:, 54:118].max())
print(X_other_information_train[:, 118:].min()," ",X_other_information_train[:, 118:].max())

print(X_other_information_test[:, :54].min()," ",X_other_information_test[:, :54].max())
print(X_other_information_test[:, 54:118].min()," ",X_other_information_test[:, 54:118].max())
print(X_other_information_test[:, 118:].min()," ",X_other_information_test[:, 118:].max())

_bus_data = case118['bus']
_gen_data_ = case118['gen']
# 母线类型
_bus_types = _bus_data[:, 1]
Vm_actio_min = []
Vm_actio_max = []
# 根据母线类型填充电压幅值
for _temp_index,_bus_type in enumerate(_bus_types):
    if _bus_type == 2 or _bus_type == 3:
        Vm_actio_min.append(_bus_data[_temp_index,-1])
        Vm_actio_max.append(_bus_data[_temp_index,-2])
Pg_actio_max = _gen_data_[:,8]/100
Pg_actio_min = _gen_data_[:,9]/100

# 转换为张量
Pg_min_tensor = torch.tensor(Pg_actio_min, device=device)  # shape: [54]
Pg_max_tensor = torch.tensor(Pg_actio_max, device=device)  # shape: [54]
Vm_min_tensor = torch.tensor(Vm_actio_min, device=device)  # shape: [54]
Vm_max_tensor = torch.tensor(Vm_actio_max, device=device)  # shape: [54]

def power_flow_equations_batch(case118, state, action, q, u, delta, q_u_delta, balance_theta):
    """
    批量化版本的power_flow_equations函数
    所有输入张量的第一维都是batch_size
    """
    bus_data = case118['bus']
    gen_data_ = case118['gen']
    branch_data = case118['branch']
    gencost_data = case118['gencost']

    num_buses = bus_data.shape[0]
    num_gens = gen_data_.shape[0]
    batch_size = state.shape[0]

    # 计算Ybus（这些是常数，不依赖于批量数据）
    Ybus = calculate_ybus(branch_data, num_buses, bus_data)
    Ybus_ = calculate_ybus_(branch_data, num_buses, bus_data)
    # print(q_u_delta)
    # 批量提取q_u_delta的各个部分
    PV_Q = q_u_delta[:, :54]  # [batch_size, 54]
    PQ_V = q_u_delta[:, 54:118]  # [batch_size, 64]
    PQV_theta = q_u_delta[:, 118:]  # [batch_size, 118]

    bus_types = torch.tensor(bus_data[:, 1], dtype=torch.int32, device=state.device)
    
    is_pv = (bus_types == 2) | (bus_types == 3)
    is_pq = (bus_types == 1)
    is_excep_balance = (bus_types == 1) | (bus_types == 2)

    Q_load_values = action[:, num_gens:]  # [batch_size, num_buses]
    PQ_V_values = PQ_V  # [batch_size, num_pq_buses]

    
    pv_bus_indices = torch.where(is_pv)[0]
    pq_bus_indices = torch.where(is_pq)[0]
    excep_balance_indices = torch.where(is_excep_balance)[0]
    
    num_pv_buses = len(pv_bus_indices)
    num_pq_buses = len(pq_bus_indices)

    # 构建Vm_combined张量 [batch_size, num_buses]
    Vm_combined = torch.zeros(batch_size, num_buses, dtype=torch.float32, device=state.device)
    
    # 为PV节点赋值
    pv_Q_load = Q_load_values[:, :num_pv_buses]
    Vm_combined[:, pv_bus_indices] = pv_Q_load
    Vm_combined[:, pq_bus_indices] = PQ_V_values   
    voltage_tensor = torch.polar(Vm_combined, PQV_theta)

    # 发电机相关计算
    gen_buses = torch.tensor(gen_data_[:, 0] - 1, dtype=torch.int64, device=state.device)
    Pg_all = action[:, :54]  # [batch_size, 54]
    Qg_all = PV_Q  # [batch_size, 54]

    # 创建每个节点的发电功率 [batch_size, num_buses]
    Pg_per_bus = torch.zeros(batch_size, num_buses, dtype=torch.float32, device=state.device)
    Qg_per_bus = torch.zeros(batch_size, num_buses, dtype=torch.float32, device=state.device)
    
    # 使用scatter_进行批量赋值
    for i in range(batch_size):
        Pg_per_bus[i].index_put_([gen_buses], Pg_all[i])
        Qg_per_bus[i].index_put_([gen_buses], Qg_all[i])
    
    P_load = state[:, :num_buses]  # [batch_size, num_buses]
    Q_load = state[:, num_buses:2*num_buses]  # [batch_size, num_buses]

    # 计算节点注入功率 [batch_size, num_buses]
    # 注意：Ybus是[bus, bus]，需要扩展维度进行批量矩阵乘法
    V_conjugate = torch.conj(voltage_tensor)
    # 批量矩阵乘法：Ybus [bus, bus] * voltage_tensor [batch, bus]
    # 需要将Ybus扩展到batch维度
    Ybus_expanded = Ybus.unsqueeze(0).expand(batch_size, -1, -1)
    # 执行批量矩阵乘法
    V_conjugate_sum = torch.bmm(Ybus_expanded, voltage_tensor.unsqueeze(-1)).squeeze(-1)
    S_injection = V_conjugate * V_conjugate_sum
    
    P_injection_ = Pg_per_bus - P_load - torch.real(S_injection)
    Q_injection_ = Qg_per_bus - Q_load + torch.imag(S_injection)

    P_injection = P_injection_[:, excep_balance_indices]
    Pg_all_new = Pg_all.clone()
    bus_29_power = (P_load + torch.real(S_injection))[:, 29:30]  # [batch_size, 1]
    Pg_all_new[:, 29:30] = bus_29_power

    Q_injection = Q_injection_[:, pq_bus_indices]
    
    # 计算PV节点的真实无功 [batch_size, num_pv_buses]
    Qg_all_true = (Q_load - torch.imag(S_injection))[:, pv_bus_indices]

    P_balance = torch.mean(torch.square(P_injection), dim=1)  # 对每个样本的节点求和
    Q_balance = torch.mean(torch.square(Q_injection), dim=1)

    # 计算成本损失
    a = torch.tensor(gencost_data[:, 4], dtype=torch.float32, device=state.device)
    b = torch.tensor(gencost_data[:, 5], dtype=torch.float32, device=state.device)
    c = torch.tensor(gencost_data[:, 6], dtype=torch.float32, device=state.device)
    
    scaled_Pg = 100 * Pg_all_new
    # 扩展系数以匹配批量维度
    a_expanded = a.unsqueeze(0).expand(batch_size, -1)
    b_expanded = b.unsqueeze(0).expand(batch_size, -1)
    c_expanded = c.unsqueeze(0).expand(batch_size, -1)
    
    cost_loss = torch.sum(a_expanded * torch.square(scaled_Pg) + b_expanded * scaled_Pg + c_expanded, dim=1)

    cost_loss_ = a_expanded * torch.square(scaled_Pg) + b_expanded * scaled_Pg + c_expanded

    # 计算无功越限损失
    Q_min = torch.tensor(gen_data_[:, 4] / 100, dtype=torch.float32, device=state.device)
    Q_max = torch.tensor(gen_data_[:, 3] / 100, dtype=torch.float32, device=state.device)
    
    Q_min_expanded = Q_min.unsqueeze(0).expand(batch_size, -1)
    Q_max_expanded = Q_max.unsqueeze(0).expand(batch_size, -1)
    
    Qg_violations_upper = torch.maximum(Qg_all_true - Q_max_expanded, torch.tensor(0.0, device=state.device))
    Qg_violations_lower = torch.maximum(Q_min_expanded - Qg_all_true, torch.tensor(0.0, device=state.device))
    total_reactive_loss = torch.sum(Qg_violations_upper + Qg_violations_lower, dim=1)

    # 计算有功越限损失
    P_min = torch.tensor(gen_data_[:, 9] / 100, dtype=torch.float32, device=state.device)
    P_max = torch.tensor(gen_data_[:, 8] / 100, dtype=torch.float32, device=state.device)
    
    P_min_expanded = P_min.unsqueeze(0).expand(batch_size, -1)
    P_max_expanded = P_max.unsqueeze(0).expand(batch_size, -1)
    
    Pg_violations_upper = torch.maximum(Pg_all_new - P_max_expanded, torch.tensor(0.0, device=state.device))
    Pg_violations_lower = torch.maximum(P_min_expanded - Pg_all_new, torch.tensor(0.0, device=state.device))
    total_active_loss = torch.sum(Pg_violations_upper + Pg_violations_lower, dim=1)/num_buses

    # 计算电压越限损失
    V_min = torch.ones(batch_size, bus_data[:, -1].shape[0], device=state.device) * 0.94
    V_max = torch.ones(batch_size, bus_data[:, -1].shape[0], device=state.device) * 1.06
        
    V_magnitudes = torch.abs(voltage_tensor)

    V_violations_upper = torch.maximum(V_magnitudes - V_max, torch.tensor(0.0, device=state.device))
    V_violations_lower = torch.maximum(V_min - V_magnitudes, torch.tensor(0.0, device=state.device))
    total_voltage_loss = torch.sum(V_violations_upper + V_violations_lower, dim=1)

    arr = branch_data[:, [0, 1, 5, 7]]
    keys = np.array([tuple(sorted(pair)) for pair in arr[:, :2]])
    unique_keys, indices = np.unique(keys, axis=0, return_inverse=True)

    sums = np.zeros(len(unique_keys))
    ratios = np.zeros(len(unique_keys))

    for i in range(len(arr)):
        from_bus, to_bus, rateA, ratio = arr[i]
        ratio = 1.0 if ratio == 0 else ratio
        sums[indices[i]] += rateA
        ratios[indices[i]] = ratio
        
    result = np.column_stack((unique_keys, sums, ratios/2))

    from_buses = torch.tensor(result[:, 0] - 1, dtype=torch.int64, device=state.device)
    to_buses = torch.tensor(result[:, 1] - 1, dtype=torch.int64, device=state.device)
    line_limits = torch.tensor(result[:, 2] / 100.0, dtype=torch.float32, device=state.device)
    
    # 扩展线路限制到批量维度
    line_limits_expanded = line_limits.unsqueeze(0).expand(batch_size, -1)

    # 计算线路潮流 [batch_size, num_lines]
    V_from = voltage_tensor[:, from_buses] 
    V_to = voltage_tensor[:, to_buses]
    Y_ij = Ybus_[from_buses, to_buses]
    # 扩展Y_ij到批量维度
    Y_ij_expanded = Y_ij.unsqueeze(0).expand(batch_size, -1)
    I_ij = Y_ij_expanded * (V_from - V_to)
    flows = torch.conj(V_from) * I_ij
    flow_magnitudes = torch.abs(flows)
    
    line_violations = torch.maximum(flow_magnitudes - line_limits_expanded, torch.tensor(0.0, device=state.device))
    total_line_loss = torch.sum(line_violations, dim=1)

    # 计算MSE损失
    mse_q = torch.mean(torch.square(PV_Q - q), dim=1)
    mse_delta = torch.mean(torch.square(PQV_theta - delta), dim=1)
    mse_u = torch.mean(torch.square(PQ_V - u), dim=1)

    # 计算平衡节点角度损失
    balance_theta_loss = torch.square(PQV_theta[:, 68] - balance_theta)

    return (torch.sqrt(P_balance/num_gens), 
            torch.sqrt(Q_balance/num_gens), 
            balance_theta_loss,
            cost_loss, 
            total_active_loss/num_gens, 
            total_reactive_loss/num_gens, 
            total_voltage_loss/num_buses, 
            total_line_loss/batch_size, 
            mse_q, 
            mse_delta, 
            mse_u)

class DiffusionDataset(Dataset):
    """扩散模型数据集"""
    def __init__(self, X_in, X_con, X_other_information, T, bar_alpha, mean=0, std=1):
        self.X_in = X_in.cpu().numpy() if torch.is_tensor(X_in) else X_in
        self.X_con = X_con.cpu().numpy() if torch.is_tensor(X_con) else X_con
        self.X_other_information = X_other_information.cpu().numpy() if torch.is_tensor(X_other_information) else X_con
        self.T = T
        self.bar_alpha = bar_alpha
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X_in)
    
    def __getitem__(self, idx):
        t = np.random.randint(0, self.T)
        y_0 = self.X_in[idx]
        x = self.X_con[idx]
        z = self.X_other_information[idx]
        
        sqrt_alpha = np.sqrt(self.bar_alpha[t])
        sqrt_beta = np.sqrt(1 - self.bar_alpha[t])
        
        noise = np.random.normal(self.mean, self.std, size=y_0.shape)
        y_t = y_0 * sqrt_alpha + noise * sqrt_beta
        
        return (
            torch.FloatTensor(y_t),
            torch.tensor([t], dtype=torch.float32),
            torch.FloatTensor(x),
            torch.FloatTensor(noise),
            torch.FloatTensor(y_0),
            torch.FloatTensor(z)
        )

# DPPM训练类
class DPPMTrainer:
    def __init__(self, model1, model2, T=1000, beta_min=1e-4, beta_max=0.02):
        self.model_DDPM = model1
        self.model_PFM = model2
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max        
        # 设置噪声调度
        self._setup_noise_schedule()
        
    def _setup_noise_schedule(self):
        """设置噪声调度"""
        t = np.arange(1, self.T + 1)
        self.beta = self.beta_min + (t - 1) / (self.T - 1) * (self.beta_max - self.beta_min)
        self.alpha = 1 - self.beta
        self.bar_alpha = np.cumprod(self.alpha)
        self.bar_beta = np.sqrt(1 - self.bar_alpha)
        
        # 转换为PyTorch张量
        self.beta = torch.FloatTensor(self.beta).to(device)
        self.alpha = torch.FloatTensor(self.alpha).to(device)
        self.bar_alpha = torch.FloatTensor(self.bar_alpha).to(device)
        self.bar_beta = torch.FloatTensor(self.bar_beta).to(device)

    def sample(self, n, z_cons, t0=0):
        """采样函数"""
        with torch.no_grad():
            # 初始化噪声
            z_samples = torch.normal(0, 1, size=(n, self.model_DDPM.act_dim)).to(device)
            z_cons = torch.FloatTensor(z_cons[:n]).to(device)
            
            for t in tqdm(range(t0, self.T), ncols=0, desc="采样"):
                current_t = self.T - t - 1
                bt = torch.full((n, 1), current_t, dtype=torch.float32).to(device)
                
                # 计算系数
                sqrt_alpha_t = torch.sqrt(self.alpha[current_t])
                beta_t = self.beta[current_t]
                bar_beta_t = torch.sqrt(1 - self.bar_alpha[current_t])
                
                # 去噪步骤
                eps_theta = self.model_DDPM(z_samples, bt, z_cons)
                z_samples = (z_samples - (beta_t / bar_beta_t) * eps_theta) / sqrt_alpha_t
                
                # 添加噪声（t > 0 时）
                if current_t > 0:
                    sigma_t = torch.sqrt(self.beta[current_t])
                    z_samples += torch.randn_like(z_samples) * sigma_t
            
            # 裁剪到[-1, 1]
            if scaler_flag == 1:
                z_samples = torch.clamp(z_samples, -1, 1)            
            else:
                z_samples[:,:54] = torch.clamp(z_samples[:,:54], Pg_min_tensor, Pg_max_tensor)    
                z_samples[:,54:] = torch.clamp(z_samples[:,54:], Vm_min_tensor, Vm_max_tensor)  
                
            return z_samples.cpu().numpy()

    def DDIM_sample(self, n, z_cons, num_steps=10, ddim_eta=0.0, t0=0, use_tqdm=False,
                    pre_noise=None, t_t=None, y_t=None, return_numpy=True):
        """
        DDIM采样函数 - 固定总步数版本
        每个样本都采样num_steps步，从各自的t_t开始到0结束
        """
        if return_numpy:
            with torch.no_grad():
                z_samples = self._ddim_sample_impl(
                    n, z_cons, num_steps, ddim_eta, t0, use_tqdm,
                    pre_noise, t_t, y_t, training=False
                ).cpu().numpy()
                if scaler_flag == 1:
                    z_samples = np.clip(z_samples, -1, 1)            
                else:
                    z_samples[:,:54] = np.clip(z_samples[:,:54], Pg_min_tensor.cpu().numpy(), Pg_max_tensor.cpu().numpy())    
                    z_samples[:,54:] = np.clip(z_samples[:,54:], Vm_min_tensor.cpu().numpy(), Vm_max_tensor.cpu().numpy())  
                return z_samples
        else:
            # 训练模式，需要保留梯度
            return self._ddim_sample_impl(
                n, z_cons, num_steps, ddim_eta, t0, use_tqdm,
                pre_noise, t_t, y_t, training=True
            )

    def _ddim_sample_impl(self, n, z_cons, num_steps=10, ddim_eta=0.0, t0=0, 
                        use_tqdm=True, pre_noise=None, t_t=None, y_t=None, 
                        training=False):
        """
        DDIM采样的内部实现
        training: 是否处于训练模式（需要保留梯度）
        """
        # 确保z_cons是张量且维度正确
        if not isinstance(z_cons, torch.Tensor):
            z_cons = torch.FloatTensor(z_cons).to(device)
        z_cons = z_cons[:n].to(device)
        
        # 1. 初始化：处理预计算噪声
        if pre_noise is not None and t_t is not None and y_t is not None:
            # 验证输入
            assert pre_noise.shape == y_t.shape, f"pre_noise形状{pre_noise.shape}与y_t形状{y_t.shape}不匹配"
            assert pre_noise.shape[0] == n, f"pre_noise样本数{pre_noise.shape[0]}与n={n}不匹配"
            
            # 转换为张量
            if not isinstance(y_t, torch.Tensor):
                y_t = torch.FloatTensor(y_t).to(device)
            if not isinstance(pre_noise, torch.Tensor):
                pre_noise = torch.FloatTensor(pre_noise).to(device)
            
            # 处理t_t
            if isinstance(t_t, (int, float)):
                t_t = torch.full((n,), t_t, dtype=torch.long).to(device)
            elif not isinstance(t_t, torch.Tensor):
                t_t = torch.tensor(t_t, dtype=torch.long).to(device)
            
            # 重要：在训练模式下，使用detach避免梯度传播
            if training:
                z_samples = y_t.clone().detach().requires_grad_(False)
            else:
                z_samples = y_t.clone().to(device)
            
            # 检查是否有起始时间步小于0的情况
            if (t_t < 0).any():
                print(f"警告：有些样本的起始时间步小于等于0: {t_t[t_t <= 0]}")
                
        else:
            # 从完全噪声开始
            z_samples = torch.randn(n, self.model_DDPM.act_dim).to(device)
            t_t = torch.full((n,), self.T - 1, dtype=torch.long).to(device)
        
        # 2. 为每个样本创建独立的时间步序列
        all_timesteps = []
        for i in range(n):
            start_t = t_t[i].item()
            
            # 如果起始时间步已经是0，则不需要采样
            if start_t <= 0:
                # 创建空的时间步序列
                all_timesteps.append(np.array([], dtype=int))
                continue
            
            # 从start_t到0（不包括0），均匀选择num_steps个时间步
            timesteps = np.linspace(start_t, 0, num_steps+1).astype(int)
            timesteps = timesteps[:-1]  # 去掉最后一个0
            
            # 确保时间步序列是递减的且没有重复
            timesteps = np.unique(timesteps)[::-1]  # 反转确保递减
            
            all_timesteps.append(timesteps)
        
        # 3. 找出最大的步数用于进度条
        max_steps = max([len(steps) for steps in all_timesteps])
        
        # 4. 创建一个标记数组，记录哪些样本已经完成采样
        completed = torch.zeros(n, dtype=torch.bool).to(device)
        
        # 对于起始时间步<=0的样本，标记为已完成
        for i in range(n):
            if t_t[i] <= 0:
                completed[i] = True
        
        # 5. DDIM采样循环
        if use_tqdm:
            progress_bar = tqdm(total=max_steps, ncols=0, desc="DDIM采样")
        
        for step_idx in range(max_steps):
            # 确定当前步要处理的样本
            active_mask = torch.zeros(n, dtype=torch.bool).to(device)
            current_t_list = torch.zeros(n, dtype=torch.long).to(device)
            t_next_list = torch.zeros(n, dtype=torch.long).to(device)
            
            for i in range(n):
                if step_idx < len(all_timesteps[i]) and not completed[i]:
                    active_mask[i] = True
                    current_t_list[i] = all_timesteps[i][step_idx]
                    t_next_list[i] = all_timesteps[i][step_idx+1] if step_idx < len(all_timesteps[i])-1 else 0
            
            # 如果没有活跃的样本，跳出循环
            if not active_mask.any():
                if use_tqdm:
                    progress_bar.update(max_steps - step_idx)
                break
            
            # 只处理活跃的样本
            active_indices = torch.where(active_mask)[0]
            
            if len(active_indices) == 0:
                if use_tqdm:
                    progress_bar.update(1)
                continue
            
            # 提取活跃样本的数据
            z_samples_active = z_samples[active_indices]
            z_cons_active = z_cons[active_indices]
            current_t_active = current_t_list[active_indices]
            t_next_active = t_next_list[active_indices]
            
            # 转换时间步为张量
            bt = current_t_active.float().unsqueeze(1).to(device)  # [active_n, 1]
            
            # 6. 计算当前时间步的系数
            alpha_bar_t = self.bar_alpha[current_t_active.cpu().numpy()]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).unsqueeze(1).to(device)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t).unsqueeze(1).to(device)
            
            # 7. 使用模型预测噪声
            if step_idx == 0 and pre_noise is not None and not training:
                # 在非训练模式下，如果有预计算噪声，则使用
                eps_theta = torch.zeros_like(z_samples_active).to(device)
                for idx, original_idx in enumerate(active_indices):
                    eps_theta[idx] = pre_noise[original_idx]
            else:
                # 调用模型预测噪声
                if training:
                    # 训练模式下，确保梯度传播
                    eps_theta = self.model_DDPM(z_samples_active.detach(), bt, z_cons_active)
                else:
                    eps_theta = self.model_DDPM(z_samples_active, bt, z_cons_active)
            
            # 8. 估计原始数据x0
            pred_x0 = (z_samples_active - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
            
            # 9. 计算下一个时间步的系数
            alpha_bar_t_next = self.bar_alpha[t_next_active.cpu().numpy()]
            sqrt_alpha_bar_t_next = torch.sqrt(alpha_bar_t_next).unsqueeze(1).to(device)
            
            # 10. 计算方差σ_t (DDIM公式)
            sigma_t = torch.zeros(len(active_indices), 1).to(device)
            if ddim_eta != 0:
                for idx, t_next in enumerate(t_next_active):
                    if t_next > 0:
                        current_t_val = current_t_active[idx].item()
                        if current_t_val > 0 and alpha_bar_t[idx] > 0:
                            sigma_t[idx] = ddim_eta * torch.sqrt(
                                (1 - alpha_bar_t_next[idx]) / (1 - alpha_bar_t[idx]) * 
                                (1 - alpha_bar_t[idx] / alpha_bar_t_next[idx])
                            )
            
            # 11. DDIM更新公式
            direction_weight = torch.sqrt(torch.clamp(1 - alpha_bar_t_next.unsqueeze(1) - sigma_t**2, min=0))
            z_samples_next = sqrt_alpha_bar_t_next * pred_x0 + direction_weight * eps_theta
            
            # 12. 添加随机噪声
            if ddim_eta > 0 and not training:  # 训练时不添加随机噪声
                for idx, t_next in enumerate(t_next_active):
                    if sigma_t[idx] > 0 and t_next > 0:
                        z_samples_next[idx] = z_samples_next[idx] + sigma_t[idx] * torch.randn_like(z_samples_next[idx])
            
            # 13. 对于t_next=0的情况，直接使用预测的x0
            for idx, t_next in enumerate(t_next_active):
                if t_next == 0:
                    z_samples_next[idx] = pred_x0[idx]
                    # 标记该样本已完成
                    original_idx = active_indices[idx].item()
                    completed[original_idx] = True
            
            # 14. 更新活跃样本的状态
            new_z_samples = z_samples.clone()
            new_z_samples[active_indices] = z_samples_next
            z_samples = new_z_samples
            
            if use_tqdm:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'active': len(active_indices),
                    'completed': completed.sum().item(),
                    'min_t': current_t_active.min().item() if len(current_t_active) > 0 else 0
                })
        
        if use_tqdm:
            progress_bar.close()
        
        return z_samples

    def DDPM_pre_train_epoch(self, dataloader, optimizer_dppm, epoch, total_epochs):
        """训练一个epoch，使用同一个优化器"""
        total_loss_1 = 0

        for batch_idx, (y_t, t, x, noise, y_0, z) in enumerate(dataloader):
            y_t = y_t.to(device)
            t = t.to(device)
            x = x.to(device)
            noise = noise.to(device)
            z = z.to(device)
            y_0 = y_0.to(device)
            torch.cuda.empty_cache()

            # === 阶段1: 训练DDPM的基础损失 ===
            # DPPM 不考虑物理信息预训练
            self.model_DDPM.train()
            optimizer_dppm.zero_grad()                
            pred_noise = self.model_DDPM(y_t, t, x)
            loss_1 = F.mse_loss(pred_noise, noise)
            loss_1.backward()
            optimizer_dppm.step()
            total_loss_1 += loss_1.item()

        num_batches = len(dataloader)
        avg_loss_1 = total_loss_1 / num_batches if num_batches > 0 else 0

        return avg_loss_1

    def PFM_pre_train_epoch(self, dataloader, optimizer_dppm, epoch, total_epochs):
        total_loss_2 = 0

        for batch_idx, (y_t, t, x, noise, y_0, z) in enumerate(dataloader):
            y_t = y_t.to(device)
            t = t.to(device)
            x = x.to(device)
            noise = noise.to(device)
            z = z.to(device)
            y_0 = y_0.to(device)
            torch.cuda.empty_cache()

            # === 阶段2: 训练pinn_pf ===
            self.model_PFM.train()
            optimizer_dppm.zero_grad()                
            # PINN PF 不考虑物理信息预训练
            losses = self.compute_PINN_loss(x_input=y_0, y_input=x, other_variable=z, epoch=epoch, penalty_coefficient=1)
            loss_2 = losses[0]
            loss_2.backward()
            optimizer_dppm.step()
            total_loss_2 += loss_2.item()

        # 计算平均损失
        num_batches = len(dataloader)
        avg_loss_2 = total_loss_2 / num_batches if num_batches > 0 else 0

        return avg_loss_2
    
    def PFM_PINN_finetune_epoch(self, dataloader, optimizer_dppm, epoch, total_epochs):
        total_loss_3 = 0

        for batch_idx, (y_t, t, x, noise, y_0, z) in enumerate(dataloader):
            y_t = y_t.to(device)
            t = t.to(device)
            x = x.to(device)
            noise = noise.to(device)
            z = z.to(device)
            y_0 = y_0.to(device)
            torch.cuda.empty_cache()

            self.model_PFM.train()    
            optimizer_dppm.zero_grad()   

            # PINN PF 考虑物理信息微调
            losses = self.compute_PINN_loss(x_input=y_0, y_input=x, other_variable=z, epoch=epoch, penalty_coefficient=1)
            loss_3 = (0 * losses[0] + 1 * losses[1] + 1 * losses[2] + 1 * losses[3] + 
                    0 * losses[4] + 0 * losses[5] + 0 * losses[6] + 0 * losses[7] + 
                    0 * losses[8] + 0 * losses[9] + 0 * losses[10] + 0 * losses[11])
            loss_3.backward()
            optimizer_dppm.step()
            total_loss_3 += loss_3.item()

        # 计算平均损失
        num_batches = len(dataloader)
        avg_loss_3 = total_loss_3 / num_batches if num_batches > 0 else 0

        return avg_loss_3

    def Combined_finetune_epoch(self, dataloader, optimizer_dppm, epoch, total_epochs):
        """训练一个epoch，使用同一个优化器"""
        total_loss_4 = 0

        for batch_idx, (y_t, t, x, noise, y_0, z) in enumerate(dataloader):
            y_t = y_t.to(device)
            t = t.to(device)
            x = x.to(device)
            noise = noise.to(device)
            z = z.to(device)
            y_0 = y_0.to(device)
            torch.cuda.empty_cache()

            self.model_DDPM.train() 
            self.model_PFM.train()  
            optimizer_dppm.zero_grad() 
            
            # 优化PINN部分                    
            pred_noise = self.model_DDPM(y_t, t, x)
            
            # 使用新的采样函数，training=True
            y_samples = self._ddim_sample_impl(
                y_t.shape[0], x, num_steps=5, ddim_eta=0.0, t0=0, 
                use_tqdm=False, pre_noise=pred_noise, t_t=t, y_t=y_t, training=True
            )
            
            # 计算损失
            losses = self.compute_PINN_loss(
                x_input=y_samples, 
                y_input=x, 
                other_variable=z, 
                epoch=epoch, 
                penalty_coefficient=1
            )
            
            # 注意：这里可能需要调整权重
            loss_4 = (0 * losses[0] + 0 * losses[1] + 0 * losses[2] + 0 * losses[3] + 
                    0 * losses[4] + 1 * losses[5] + 1 * losses[6] + 1 * losses[7] + 
                    0 * losses[8] + 0 * losses[9] + 0 * losses[10] + 0 * losses[11])     
            
            # DDPM损失
            loss_ddpm = F.mse_loss(pred_noise, noise)
            # print('loss_ddpm', loss_ddpm)
            
            # 总损失
            loss_4 = 0.01*loss_4 + 1*loss_ddpm

            # 反向传播
            loss_4.backward()
            optimizer_dppm.step()
            
            total_loss_4 += loss_4.item()

            if batch_idx > 1:
                break
        
        return total_loss_4 / len(dataloader)

    def compute_PINN_loss(self, x_input, y_input, other_variable, epoch, penalty_coefficient):
        q_u_delta = self.model_PFM(x_input, y_input)
        xent_loss = F.mse_loss(q_u_delta, other_variable)
        # 批量提取变量
        samples = x_input.shape[0]
        
        # 批量提取其他变量
        q = other_variable[:, :54]  # [batch_size, 54]
        u = other_variable[:, 54:118]  # [batch_size, 64]
        delta = other_variable[:, 118:]  # [batch_size, 118]
        balance_theta = other_variable[:, 118 + 68:118 + 69].squeeze()  # [batch_size] 或 [batch_size, 1]
        
        # 批量调用power_flow_equations
        (grad_P_balance, grad_Q_balance, grad_theta_balance, grad_cost_loss, 
        grad_active_loss, grad_reactive_loss, grad_voltage_loss, grad_line_loss, 
        grad_mse_q, grad_mse_delta, grad_mse_u) = power_flow_equations_batch(
            case118, y_input, x_input, q, u, delta, q_u_delta, balance_theta)
        
        # 计算平均损失
        return (penalty_coefficient * xent_loss,
                penalty_coefficient * grad_P_balance.mean(),
                penalty_coefficient * grad_Q_balance.mean(),
                grad_theta_balance.mean(),
                penalty_coefficient * grad_cost_loss.mean(),
                penalty_coefficient * grad_active_loss.mean(),
                penalty_coefficient * grad_reactive_loss.mean(),
                penalty_coefficient * grad_voltage_loss.mean(),
                penalty_coefficient * grad_line_loss.mean(),
                penalty_coefficient * grad_mse_q.mean(),
                penalty_coefficient * grad_mse_delta.mean(),
                penalty_coefficient * grad_mse_u.mean())

# 学习率调度器
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, peak_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # 线性warmup
            lr = self.peak_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # 余弦衰减
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr

# 学习率调度器
class LinearDecreaseScheduler:
    def __init__(self, optimizer, total_epochs, peak_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1    
        # 线性warmup
        progress = self.current_epoch / self.total_epochs
        lr = self.peak_lr - (self.peak_lr - self.min_lr) * (progress)      
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr            
        return lr

# 学习率调度器
class FlexibleFiveStageScheduler:
    def __init__(self, optimizer, stage_epochs):
        """
        五段线性学习率调度器（灵活配置）
        
        参数:
            optimizer: 优化器
            stage_epochs: 每个阶段的epoch数列表，如 [10, 20, 20, 30, 20]
        """
        self.optimizer = optimizer
        self.stage_epochs = stage_epochs
        self.current_epoch = 0
        
        # 定义五段线性化的学习率变化
        self.stage_lrs = [
            (0, 0.001),          # 第一阶段: 0 -> 0.001
            (0.001, 0.00001),    # 第二阶段: 0.001 -> 0.00001
            (0.0001, 0.000001),    # 第三阶段: 0.001 -> 0.00001
            (0.00001, 0.000001), # 第四阶段: 0.00001 -> 0.000001
            (0.000000001, 0.0000000001) # 第五阶段: 0.000001 -> 0.0000001
        ]
        
        # 计算每个阶段的起始和结束epoch
        self.stages = []
        current_start = 0
        for i, epochs in enumerate(stage_epochs):
            stage = {
                'start_epoch': current_start + 1,
                'end_epoch': current_start + epochs,
                'start_lr': self.stage_lrs[i][0],
                'end_lr': self.stage_lrs[i][1]
            }
            self.stages.append(stage)
            current_start += epochs
        self.total_epochs = sum(stage_epochs)
        
    def step(self):
        """更新学习率"""
        self.current_epoch += 1
        
        # 查找当前epoch所在的阶段
        for stage in self.stages:
            if stage['start_epoch'] <= self.current_epoch <= stage['end_epoch']:
                # 计算当前阶段内的进度（0到1之间）
                stage_progress = (self.current_epoch - stage['start_epoch']) / \
                                 (stage['end_epoch'] - stage['start_epoch'])
                
                # 线性插值计算当前学习率
                lr = stage['start_lr'] + (stage['end_lr'] - stage['start_lr']) * stage_progress
                break
        else:
            # 如果超出最后一个阶段，使用最后的学习率
            lr = self.stages[-1]['end_lr']
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
    
    def get_current_lr(self):
        """获取当前学习率"""
        for stage in self.stages:
            if stage['start_epoch'] <= self.current_epoch <= stage['end_epoch']:
                stage_progress = (self.current_epoch - stage['start_epoch']) / \
                                 (stage['end_epoch'] - stage['start_epoch'])
                lr = stage['start_lr'] + (stage['end_lr'] - stage['start_lr']) * stage_progress
                return lr
        return self.stages[-1]['end_lr']

# 电力系统分析函数（NumPy版本，保持兼容性）
def calculate_ybus_numpy(branch_data, num_buses, bus_data):
    """计算导纳矩阵 Ybus - NumPy版本"""
    Ybus = np.zeros((num_buses, num_buses), dtype=np.complex64)
    
    for branch in branch_data:
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        resistance = branch[2]
        reactance = branch[3]
        b = branch[4]
        
        impedance = resistance + 1j * reactance
        Y = 1 / impedance
        
        if branch[-5] == 0:
            ratio = 1.0
            angle_rad = np.deg2rad(branch[-4])
        else:
            ratio = branch[-5]
            angle_rad = np.deg2rad(branch[-4])
        
        Ybus[from_bus, from_bus] += (Y + 1j * (b / 2)) / ratio**2
        Ybus[to_bus, to_bus] += (Y + 1j * (b / 2))
        
        Ybus[from_bus, to_bus] -= Y / ratio * np.exp(1j * angle_rad)
        Ybus[to_bus, from_bus] -= Y / ratio * np.exp(-1j * angle_rad)
    
    for i in range(num_buses):
        Gs = bus_data[i][4]
        Bs = bus_data[i][5]
        Ybus[i, i] += Gs / 100 + 1j * Bs / 100
    
    return Ybus

def AC_optimal_power_flow_equations_evaluation(case118, state, action, q, u, delta, sample_idx=None):
    """交流最优潮流方程评估 - NumPy版本"""
    # 提取数据
    bus_data = case118['bus']
    gen_data_ = case118['gen']
    branch_data = case118['branch']
    gencost_data = case118['gencost']

    # 母线数量
    num_buses = bus_data.shape[0]
    num_gens = gen_data_.shape[0]
    num_branchs = branch_data.shape[0]

    # 计算Ybus
    Ybus = (calculate_ybus(branch_data, num_buses, bus_data)).cpu().detach().numpy()

    # 初始化功率平衡方程
    P_balance = 0
    Q_balance = 0
    total_line_loss = 0
    total_voltage_loss = 0
    total_active_loss = 0
    total_reactive_loss = 0
    total_cost_loss = 0

    # 母线类型
    bus_types = bus_data[:, 1]

    # 初始化电压幅值组合
    Vm_combined = []
    PQ_index = 0
    Q_load_index = 0

    # 根据母线类型填充电压幅值
    for bus_type in bus_types:
        if bus_type == 2 or bus_type == 3:
            Vm_combined.append(action[num_gens:][Q_load_index])
            Q_load_index += 1
        elif bus_type == 1:
            Vm_combined.append(u[PQ_index])
            PQ_index += 1
    
    Vm_combined = np.array(Vm_combined)
    PQV_theta = np.array(delta)

    # 计算复数电压
    voltage_tensor = Vm_combined * np.exp(1j * PQV_theta)

    # 实功和无功功率注入
    for i in range(num_buses):
        # 母线负荷
        P_load = state[i]  # 负荷有功功率
        Q_load = state[i + num_buses]  # 负荷无功功率

        # 发电机功率输出
        gen_mask = gen_data_[:, 0] == (i + 1)
        Pg = np.sum(action[:NUM_GENERATORS][gen_mask])  # 发电机有功功率
        Qg = np.sum(q[gen_mask])  # 发电机无功功率

        # 电压计算
        V_conjugate_sum = np.sum(Ybus[i, :] * voltage_tensor)
          
        P_injection = Pg - P_load - np.real(np.conj(voltage_tensor[i]) * V_conjugate_sum)  
        Q_injection = Qg - Q_load + np.imag(np.conj(voltage_tensor[i]) * V_conjugate_sum) 
       
        # 更新平衡
        P_balance += P_injection ** 2
        Q_balance += Q_injection ** 2

        # 计算发电机成本损失
        cost_params = gencost_data[gen_mask]
        if cost_params.shape[0] > 0:
            a = cost_params[:, 4]  # 二次成本系数
            b = cost_params[:, 5]  # 一次成本系数
            c = cost_params[:, 6]  # 固定成本
            
            gen_power = 100 * action[:NUM_GENERATORS][gen_mask]
            cost_loss = np.sum(a * np.square(gen_power) + b * gen_power + c)
            total_cost_loss += cost_loss

        # 计算无功功率限制
        Q_min = np.sum(gen_data_[gen_mask, 4]) / 100  # 最小无功输出
        Q_max = np.sum(gen_data_[gen_mask, 3]) / 100  # 最大无功输出
        
        if Qg > Q_max:
            reactive_loss = np.abs(Qg - Q_max)  # L1范数
            total_reactive_loss += reactive_loss
        
        elif Qg < Q_min:
            reactive_loss = np.abs(Q_min - Qg)  # L1范数
            total_reactive_loss += reactive_loss

        # 计算有功功率限制
        P_min = np.sum(gen_data_[gen_mask, 9]) / 100  # 最小有功输出
        P_max = np.sum(gen_data_[gen_mask, 8]) / 100  # 最大有功输出
        
        if Pg > P_max:
            active_loss = np.abs(Pg - P_max)  # L1范数
            total_active_loss += active_loss
        elif Pg < P_min:
            active_loss = np.abs(P_min - Pg)  # L1范数
            total_active_loss += active_loss

        # 计算电压限制
        V_min = bus_data[i, -1]  # 最小电压限制
        V_max = bus_data[i, -2]  # 最大电压限制
        V_magnitude = np.abs(voltage_tensor[i])  # 电压幅值
        
        if V_magnitude > V_max:
            voltage_loss = np.abs(V_magnitude - V_max)  # L1范数
            total_voltage_loss += voltage_loss
        elif V_magnitude < V_min:
            voltage_loss = np.abs(V_min - V_magnitude)  # L1范数
            total_voltage_loss += voltage_loss

    Ybus_ = calculate_ybus_numpy(branch_data, num_buses, bus_data)
    # 计算线路限制损失
    for branch in branch_data:
        from_bus = int(branch[0]) - 1
        to_bus = int(branch[1]) - 1
        line_limit = branch[5] / 100.0  # 线路最大潮流限制

        # 计算母线电压
        V_from = voltage_tensor[from_bus]
        V_to = voltage_tensor[to_bus]
        
        # 计算支路电流
        Y_ij = Ybus_[from_bus, to_bus]
        I_ij = Y_ij * (V_from - V_to)
        
        # 计算支路潮流
        flow = V_from * np.conj(I_ij)

        # 计算实际潮流幅值
        flow_magnitude = np.abs(flow)

        # 如果潮流超过线路限制，计算损失
        if flow_magnitude > line_limit:
            line_loss = np.abs(flow_magnitude - line_limit)  # L1范数
            total_line_loss += line_loss

    # 计算总约束数量（用于平均越限计算）
    total_constraints = num_gens + num_gens + num_buses + num_branchs
    
    return (
        np.sqrt(P_balance / num_buses),
        np.sqrt(Q_balance / num_buses),
        total_cost_loss,
        total_active_loss / num_gens,  # 有功越限平均值（按发电机数）
        total_reactive_loss / num_gens,  # 无功越限平均值（按发电机数）
        total_voltage_loss / num_buses,  # 电压越限平均值（按母线数）
        total_line_loss / num_branchs,  # 线路越限平均值（按支路数）
        (total_active_loss + total_reactive_loss + total_voltage_loss + total_line_loss) / total_constraints  # 平均越限（所有约束一起平均）
    )

def run_power_flow_pypower(case118, state, action):
    """
    使用pypower运行潮流计算，根据X_con和预测action获取q, u, delta
    并更新action中的平衡节点有功功率为潮流计算结果
    
    Parameters:
    -----------
    case118 : dict
        电网数据
    state : array
        X_con数据 [Pd, Qd] for all buses
    action : array  
        预测的动作 [Pg_gen (54), Vm_pv (54)] for generators and PV buses
        
    Returns:
    --------
    q : array
        发电机无功功率 (54,)
    u : array
        PQ节点电压幅值 (64,)
    delta : array
        所有节点电压相角 (118,)
    action_updated : array
        更新后的action，其中平衡节点有功功率已更新为潮流计算结果
    success : bool
        潮流计算是否成功
    """
    from pypower.api import runpf, ppoption
    import copy
    
    # 复制case118数据 (深拷贝字典)
    ppc = copy.deepcopy(case118)
    
    # 设置潮流计算选项
    ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)
    
    bus_data = ppc['bus']
    gen_data = ppc['gen']
    num_buses = bus_data.shape[0]
    num_gens = gen_data.shape[0]
    
    # 更新负荷数据 (Pd, Qd) - state是归一化的，需要乘以100
    for i in range(num_buses):
        bus_data[i, 2] = state[i] * 100  # Pd
        bus_data[i, 3] = state[i + num_buses] * 100  # Qd
    
    # 获取母线类型
    bus_types = bus_data[:, 1]
    
    # 更新发电机有功功率和电压幅值
    gen_index = 0
    vm_index = num_gens  # Vm values start after Pg values in action
    
    for i in range(num_buses):
        if bus_types[i] == 2 or bus_types[i] == 3:  # PV bus or Slack bus
            # 找到对应的发电机
            for g_idx in range(num_gens):
                if int(gen_data[g_idx, 0]) == i + 1:  # gen bus number matches
                    gen_data[g_idx, 1] = action[gen_index] * 100  # Pg
                    gen_data[g_idx, 5] = action[vm_index]  # Vm
                    gen_index += 1
                    vm_index += 1
                    break
    
    # 运行潮流计算
    try:
        results, success = runpf(ppc, ppopt)
        
        if not success:
            return None, None, None, None, False
        
        # 提取结果
        bus_results = results['bus']
        gen_results = results['gen']
        
        # 提取发电机无功功率 q (54,)
        q = np.zeros(num_gens)
        for g_idx in range(num_gens):
            q[g_idx] = gen_results[g_idx, 2] / 100  # Qg normalized
        
        # 提取PQ节点电压幅值 u (64,)
        pq_bus_indices = np.where(bus_types == 1)[0]
        u = np.zeros(len(pq_bus_indices))
        for idx, bus_idx in enumerate(pq_bus_indices):
            u[idx] = bus_results[bus_idx, 7]  # Vm
        
        # 提取所有节点电压相角 delta (118,)
        delta = np.zeros(num_buses)
        for i in range(num_buses):
            delta[i] = np.deg2rad(bus_results[i, 8])  # Va in radians
        
        # 更新action中的平衡节点有功功率为潮流计算结果，其他机组保持原值
        action_updated = action.copy()
        pg_updated = (action[:num_gens]).copy()  # 默认保持原action中的有功
        
        # 只更新平衡节点（slack bus）的有功功率
        for g_idx in range(num_gens):
            bus_num = int(gen_data[g_idx, 0])  # 发电机所在母线编号
            bus_idx = bus_num - 1  # 转换为0-based索引
            if bus_types[bus_idx] == 3:  # 平衡节点（slack bus）
                pg_updated[g_idx] = gen_results[g_idx, 1] / 100  # 使用潮流计算结果
                break  # 只有一个平衡节点
        
        action_updated[:num_gens] = pg_updated
        
        return q, u, delta, action_updated, True
        
    except Exception as e:
        print(f"Power flow calculation failed: {e}")
        return None, None, None, None, False


def calculate_errors(case118, X_con_test, X_pre, X_in_test, X_other_information_test, _pre_data_pinn):
    """计算各种误差"""
    pinn_p_error = 0
    pinn_q_error = 0
    pinn_cost_error = 0
    pinn_active_error = 0
    pinn_reactive_error = 0
    pinn_voltage_error = 0
    pinn_line_error = 0

    base_p_error = 0
    base_q_error = 0
    base_cost_error = 0
    base_active_error = 0
    base_reactive_error = 0
    base_voltage_error = 0
    base_line_error = 0
    base_avg_violation_error = 0 

    pinn_active_errors = []
    pinn_reactive_errors = []
    pinn_voltage_errors = []
    pinn_cost_errors = []
    pinn_avg_violation_errors = []
    
    base_active_errors = []
    base_reactive_errors = []
    base_voltage_errors = []
    base_cost_errors = []
    base_avg_violation_errors = [] 
    
    # 计数成功运行的潮流计算
    pinn_success_count = 0
    base_success_count = 0

    for i in range(X_con_test.shape[0]):
        # DDPM/PINN模型 - 使用pypower潮流计算结果，并更新平衡节点有功
        q_pinn, u_pinn, delta_pinn, action_pinn_updated, pinn_success = run_power_flow_pypower(case118, X_con_test[i,:], X_pre[i,:])
        if pinn_success and q_pinn is not None:
            _temp_p, _temp_q, _temp_cost, _temp_active, _temp_reactive, _temp_voltage, _temp_line, _temp_avg_violation = \
                AC_optimal_power_flow_equations_evaluation(
                    case118, X_con_test[i,:], action_pinn_updated, 
                    q_pinn, u_pinn, delta_pinn, sample_idx=i
                )
            pinn_p_error += _temp_p
            pinn_q_error += _temp_q
            pinn_cost_error += _temp_cost
            pinn_active_error += _temp_active
            pinn_reactive_error += _temp_reactive
            pinn_voltage_error += _temp_voltage
            pinn_line_error += _temp_line
            # 存储单个样本误差（平均值，用于表格显示）
            pinn_active_errors.append(_temp_active)
            pinn_reactive_errors.append(_temp_reactive)
            pinn_voltage_errors.append(_temp_voltage)
            pinn_cost_errors.append(_temp_cost)
            pinn_avg_violation_errors.append(_temp_avg_violation)
            pinn_success_count += 1

        # 基准模型 - 使用pypower潮流计算结果，并更新平衡节点有功
        q_base, u_base, delta_base, action_base_updated, base_success = run_power_flow_pypower(case118, X_con_test[i,:], X_in_test[i,:])
        if base_success and q_base is not None:
            _temp_p, _temp_q, _temp_cost, _temp_active, _temp_reactive, _temp_voltage, _temp_line, _temp_avg_violation = \
                AC_optimal_power_flow_equations_evaluation(
                    case118, X_con_test[i,:], action_base_updated,
                    q_base, u_base, delta_base, sample_idx=i
                )
            base_p_error += _temp_p
            base_q_error += _temp_q
            base_cost_error += _temp_cost
            base_active_error += _temp_active
            base_reactive_error += _temp_reactive
            base_voltage_error += _temp_voltage
            base_line_error += _temp_line
            base_avg_violation_error += _temp_avg_violation
            # 存储单个样本误差（平均值，用于表格显示）
            base_active_errors.append(_temp_active)
            base_reactive_errors.append(_temp_reactive)
            base_voltage_errors.append(_temp_voltage)
            base_cost_errors.append(_temp_cost)
            base_avg_violation_errors.append(_temp_avg_violation)
            base_success_count += 1

    print(f"Power flow success counts - DDPM: {pinn_success_count}/{X_con_test.shape[0]}, BASE: {base_success_count}/{X_con_test.shape[0]}")
    print()

    # 准备表格数据
    metrics = ['有功越限', '无功越限', '电压越限', '线路越限', '平均越限', '成本']
    
    if pinn_success_count > 0:
        pinn_data = [
            pinn_active_error/pinn_success_count,
            pinn_reactive_error/pinn_success_count,
            pinn_voltage_error/pinn_success_count,
            pinn_line_error/pinn_success_count,
            np.mean(pinn_avg_violation_errors),
            pinn_cost_error/pinn_success_count
        ]
    else:
        pinn_data = [None] * 6
                
    if base_success_count > 0:
        base_data = [
            base_active_error/base_success_count,
            base_reactive_error/base_success_count,
            base_voltage_error/base_success_count,
            base_line_error/base_success_count,
            np.mean(base_avg_violation_errors),
            base_cost_error/base_success_count
        ]
    else:
        base_data = [None] * 6

    # 打印表格
    print("=" * 80)
    print("电力系统误差分析结果")
    print("=" * 80)
    
    # 表头
    print(f"{'指标':<12} {'BASE':>16} {'DDPM':>16} {'DDPM vs BASE':>18}")
    print("-" * 80)
    
    # 数据行
    for i, metric in enumerate(metrics):
        base_val = base_data[i]
        pinn_val = pinn_data[i]
        
        if base_val is None:
            print(f"{metric:<12} {'N/A':>16} {'N/A':>16} {'N/A':>18}")
            continue
            
        # 格式化数值
        if metric == '成本':
            base_str = f"{base_val:.2f}"
            pinn_str = f"{pinn_val:.2f}" if pinn_val is not None else "N/A"
            # 成本指标计算百分比变化
            if pinn_val is not None:
                if base_val != 0:
                    pinn_diff = f"{((pinn_val - base_val)/abs(base_val))*100:+.2f}%"
                else:
                    pinn_diff = "N/A"
            else:
                pinn_diff = "N/A"
        else:
            base_str = f"{base_val:.6e}"
            pinn_str = f"{pinn_val:.6e}" if pinn_val is not None else "N/A"
            # 非成本指标显示N/A
            pinn_diff = "N/A"
        
        print(f"{metric:<12} {base_str:>16} {pinn_str:>16} {pinn_diff:>18}")
    
    print("=" * 80)

# 主训练流程
def main():
    # 参数设置
    state_dim = X_con_train.shape[1]
    act_dim = X_in_train.shape[1]
    batch_size = 512
    
    print(f"状态维度: {state_dim}, 动作维度: {act_dim}")
    print(f"训练样本数: {len(X_con_train)}, 测试样本数: {len(X_con_test)}")
    
    # 创建模型 
    DPPM_model = DPPM_class(act_dim=act_dim, intermediate_dim=32, state_dim=state_dim, latent_dim=2).to(device)
    trainer = DPPMTrainer(DPPM_model.DDPM,DPPM_model.PINN_PF)
    
    # 创建数据集和数据加载器
    dataset = DiffusionDataset(X_in_train_tensor, X_con_train_tensor, X_other_information_train_tensor, trainer.T, trainer.bar_alpha.cpu().numpy())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 优化器和学习率调度器
    optimizer_ddpm = optim.AdamW(DPPM_model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    
    ###
    #DDPM训练循环
    ###
    total_epochs = 4000
    warmup_epochs = 100
    peak_lr = 0.001
    scheduler_ddpm = WarmupCosineScheduler(optimizer_ddpm, warmup_epochs, total_epochs, peak_lr)
    # scheduler_ddpm = LinearDecreaseScheduler(optimizer_ddpm, total_epochs, peak_lr, min_lr=1e-8)

    best_loss = float('inf')
    losses = []

    # for epoch in tqdm(range(1, total_epochs + 1), desc='训练进度'):
    #     # 更新学习率
    #     current_lr = scheduler_ddpm.step()
    #     # 训练一个epoch
    #     avg_loss_1= trainer.DDPM_pre_train_epoch(dataloader, optimizer_ddpm, epoch, total_epochs)
    #     losses.append(avg_loss_1)   
        
    #     # 打印进度
    #     if epoch % 10 == 0:
    #         print(f'Epoch {epoch}/{total_epochs}, Loss: {avg_loss_1:.4f}, LR: {current_lr:.9f}')
            
    #         # 验证
    #         if epoch % 10 == 0:
    #             with torch.no_grad():
    #                 _temp_index = np.random.choice(len(X_con_test), 20)
    #                 z_cons_sample = X_con_test[_temp_index]
    #                 samples = trainer.sample(n=20, z_cons=z_cons_sample, t0=0)
    #                 temp_rmse = RMSE(samples, X_in_test[_temp_index])
    #                 print(f'验证损失: {temp_rmse:.4f}')

    # # # 保存模型
    # # torch.save(DPPM_model.DDPM.state_dict(), './save_model/DDPM_based_model_118_scrach.pth')
    # # print("模型已保存")        
    
    # # DPPM_model.DDPM.load_state_dict(torch.load('./save_model/DDPM_based_model_118_scrach.pth'))
    
    # ###
    # # PFM训练循环
    # ###
    # total_epochs = 40
    # warmup_epochs = 10
    # peak_lr = 0.005
    # scheduler_ddpm = WarmupCosineScheduler(optimizer_ddpm, warmup_epochs, total_epochs, peak_lr)

    # best_loss = float('inf')
    # losses = []

    # for epoch in tqdm(range(1, total_epochs + 1), desc='训练进度'):
    #     # 更新学习率
    #     current_lr = scheduler_ddpm.step()
    #     # 训练一个epoch
    #     avg_loss_2= trainer.PFM_pre_train_epoch(dataloader, optimizer_ddpm, epoch, total_epochs)
    #     losses.append(avg_loss_2)   
    #     # break

    #     # 打印进度
    #     if epoch % 1 == 0:
    #         print(f'Epoch {epoch}/{total_epochs}, Loss: {avg_loss_2:.4f}, LR: {current_lr:.9f}')
    
    # # # 保存模型
    # # torch.save(DPPM_model.PINN_PF.state_dict(), './save_model/PFM_based_model_118.pth')
    # # print("模型已保存")
    

    # # DPPM_model.PINN_PF.load_state_dict(torch.load('./save_model/PFM_based_model_118.pth'))

    # '''
    # PFM微调训练循环
    # '''
    # total_epochs = 50
    # warmup_epochs = 10
    # peak_lr = 0.0001 #0.00000001
    # scheduler_ddpm = WarmupCosineScheduler(optimizer_ddpm, warmup_epochs, total_epochs, peak_lr)

    # best_loss = float('inf')
    # losses = []

    # for epoch in tqdm(range(1, total_epochs + 1), desc='训练进度'):
    #     # 更新学习率
    #     current_lr = scheduler_ddpm.step()
    #     # 训练一个epoch
    #     avg_loss_3= trainer.PFM_PINN_finetune_epoch(dataloader, optimizer_ddpm, epoch, total_epochs)
    #     losses.append(avg_loss_3)   
    #     # 打印进度
    #     if epoch % 10 == 0:
    #         print(f'Epoch {epoch}/{total_epochs}, Loss: {avg_loss_3:.4f}, LR: {current_lr:.9f}')
    
    # # # 保存模型
    # # torch.save(DPPM_model.PINN_PF.state_dict(), './save_model/PFM_PINN_based_model_118.pth')
    # # print("模型已保存")

    # # DPPM_model.PINN_PF.load_state_dict(torch.load('./save_model/PFM_PINN_based_model_118.pth'))

    # '''
    # Combined 微调训练循环
    # '''    
    # total_epochs = 50
    # warmup_epochs = 10
    # peak_lr = 1e-6
    # # scheduler_ddpm = WarmupCosineScheduler(optimizer_ddpm, warmup_epochs, total_epochs, peak_lr, min_lr=1e-9)
    # scheduler_ddpm = LinearDecreaseScheduler(optimizer_ddpm, total_epochs, peak_lr, min_lr=1e-8)

    # best_loss = float('inf')
    # losses = []    
    # for epoch in tqdm(range(1, total_epochs + 1), desc='训练进度'):
    #     # 更新学习率
    #     current_lr = scheduler_ddpm.step()

    #     # 训练一个epoch
    #     avg_loss_4 = trainer.Combined_finetune_epoch(dataloader, optimizer_ddpm, epoch, total_epochs)
    #     losses.append(avg_loss_4)   
    #     # break
        
    #     # 打印进度
    #     if epoch % 100 == 0:
    #         print(f'Epoch {epoch}/{total_epochs}, Loss: {avg_loss_4:.4f}, LR: {current_lr:.9f}')
            
    #         # 验证
    #         if epoch % 10 == 0:
    #             with torch.no_grad():
    #                 _temp_index = np.random.choice(len(X_con_test), 20)
    #                 z_cons_sample = X_con_test[_temp_index]
    #                 samples = trainer.sample(n=20, z_cons=z_cons_sample, t0=0)
    #                 temp_rmse = RMSE(samples, X_in_test[_temp_index])
    #                 print(f'验证损失: {temp_rmse:.4f}')
    
    #     # if epoch >= 200:
    #     #     break
    # torch.save(DPPM_model.DDPM.state_dict(), './save_model/DDPM_based_model_118_finished_epoch_'+str(epoch)+'_scrach.pth')
    # torch.save(DPPM_model.PINN_PF.state_dict(), './save_model/PFM_PINN_based_model_118_finished_epoch_'+str(epoch)+'_scrach.pth')

    epoch = 50
    DPPM_model.DDPM.load_state_dict(torch.load('./save_model/DDPM_based_model_118_small_finished_epoch_'+str(epoch)+'_scrach.pth'))
    DPPM_model.PINN_PF.load_state_dict(torch.load('./save_model/PFM_PINN_based_model_118_small_finished_epoch_'+str(epoch)+'_scrach.pth'))

    # 采样
    print("开始采样...")
    start_time = time.time()
    X_pre = trainer.sample(n=len(X_con_test), z_cons=X_con_test, t0=0)
    end_time = time.time()
    print(f"Total Sampling Time: {(end_time-start_time)}" )

    if scaler_flag == 1:    
        # 反归一化
        X_in_test_ori = scaler1.inverse_transform(X_in_test)
        pre_data_ddpm_ori = scaler1.inverse_transform(X_pre)
    else:
        # 反归一化
        X_in_test_ori = X_in_test
        pre_data_ddpm_ori = X_pre
    
    if scaler_flag == 1:    
        X_con_test_inv = scaler2.inverse_transform(X_con_test)
        X_other_information_test_inv = scaler3.inverse_transform(X_other_information_test)
        _pre_data_pinn = DPPM_model.pinn_pf(torch.FloatTensor(X_pre).to(device), torch.FloatTensor(X_con_test).to(device))
        _pre_data_pinn_inv = scaler3.inverse_transform(_pre_data_pinn)
    else:
        X_con_test_inv = X_con_test
        X_other_information_test_inv = X_other_information_test
        _pre_data_pinn_inv = DPPM_model.pinn_pf(torch.FloatTensor(X_pre).to(device), torch.FloatTensor(X_con_test).to(device)).cpu().detach().numpy()

    # 计算电力系统误差
    print("\n电力系统误差分析:")
    calculate_errors(case118, X_con_test_inv, pre_data_ddpm_ori, X_in_test_ori, X_other_information_test_inv, _pre_data_pinn_inv)


if __name__ == "__main__":
    main()
