"""
Physical-parameterized correction modules for Luminance-GS++改造.

本文件把伪标签生成里的两个"自由黑箱"替换为物理参数化模块:
  - 模块 A: 色温标量 T_k -> Bradford 色适应矩阵 (替/补 自由颜色变换)。
  - 模块 B: 曝光标量 e_k × 全视角共享相机响应曲线 CRF (替/补 自由 256-LUT)。

另外提供合成退化管线 (Planckian 色偏 + 曝光增益), 产出真值 T_gt / K_gt,
用于可解释性散点图 (T_k vs T_gt, e_k vs K_gt)。

所有函数都可微、数值稳定 (逆矩阵用解析对角构造, 不对小矩阵反复数值求逆)。
仓库 batch_size 恒为 1, 因此这里以 "单视角标量 T/e -> 3x3 矩阵" 为主路径。
"""

import csv
import os
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 固定常量矩阵 (CPU 缓存, 用到时 .to(device))
# --------------------------------------------------------------------------- #
# sRGB(linear, D65) <-> CIE XYZ
_RGB2XYZ = torch.tensor(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]],
    dtype=torch.float32,
)
_XYZ2RGB = torch.inverse(_RGB2XYZ)
# Bradford 色适应锥响应矩阵
_M_A = torch.tensor(
    [[0.8951, 0.2664, -0.1614],
     [-0.7502, 1.7135, 0.0367],
     [0.0389, -0.0685, 1.0296]],
    dtype=torch.float32,
)
_M_A_inv = torch.inverse(_M_A)
# D65 白点 (Y = 1)
_XYZ_D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)

# 合成 / 预测 共用的色温区间 (Kelvin)
CCT_MIN = 1800.0
CCT_MAX = 9500.0


# --------------------------------------------------------------------------- #
# 色温 -> 白点 (Planckian locus 闭式近似, 可微)
# --------------------------------------------------------------------------- #
def cct_to_xyY(T: torch.Tensor):
    """色温 T(Kelvin) -> CIE1931 xy 色度坐标 (Kim et al. 近似, 可微)。

    若仓库别处已有 Planck->sRGB 函数, 应优先复用以保证与合成一致;
    本仓库 (CVPR 会议版) 不存在该函数, 故使用此参考实现。
    """
    T = T.clamp(1667.0, 25000.0)
    invT = 1.0 / T
    x = torch.where(
        T <= 4000.0,
        -0.2661239e9 * invT ** 3 - 0.2343589e6 * invT ** 2
        + 0.8776956e3 * invT + 0.179910,
        -3.0258469e9 * invT ** 3 + 2.1070379e6 * invT ** 2
        + 0.2226347e3 * invT + 0.240390,
    )
    y = torch.where(
        T <= 2222.0,
        -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683,
        torch.where(
            T <= 4000.0,
            -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867,
            3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483,
        ),
    )
    return x, y


def cct_to_XYZ(T: torch.Tensor) -> torch.Tensor:
    """色温 -> XYZ 白点 (Y 归一为 1)。返回 shape (..., 3)。"""
    x, y = cct_to_xyY(T)
    y = y.clamp_min(1e-6)
    X = x / y
    Y = torch.ones_like(x)
    Z = (1.0 - x - y) / y
    return torch.stack([X, Y, Z], dim=-1)


# --------------------------------------------------------------------------- #
# 模块 A: Bradford 色适应矩阵
# --------------------------------------------------------------------------- #
def bradford_correction_matrix(
    T_k: torch.Tensor, device: torch.device, inverse: bool = False
) -> torch.Tensor:
    """返回 3x3 矩阵 (sRGB-linear 空间)。

    inverse=False: 把 "光源 T_k 下的图像" 适应回 D65 (去色偏), 对应原 M_k^{-1} 的角色;
    inverse=True : 把 D65 图像投到 "光源 T_k 下" (加色偏), 对应原 M_k 的角色。

    逆通过对角阵的解析倒数构造 (D -> 1/D), 不做矩阵数值求逆。
    """
    M_A = _M_A.to(device)
    M_A_inv = _M_A_inv.to(device)
    RGB2XYZ = _RGB2XYZ.to(device)
    XYZ2RGB = _XYZ2RGB.to(device)

    XYZ_src = cct_to_XYZ(T_k).to(device).reshape(3)   # 估计光源白点
    XYZ_dst = _XYZ_D65.to(device)

    lms_src = M_A @ XYZ_src
    lms_dst = M_A @ XYZ_dst
    ratio = lms_dst / lms_src.clamp_min(1e-6)         # 去色偏 (src -> D65)
    if inverse:
        ratio = 1.0 / ratio.clamp_min(1e-6)           # 加色偏 (D65 -> src)
    D = torch.diag(ratio)

    M_cat_xyz = M_A_inv @ D @ M_A
    M_rgb = XYZ2RGB @ M_cat_xyz @ RGB2XYZ
    return M_rgb


def apply_color_adapt(
    img_rgb: torch.Tensor, T_k: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """对图像施加色适应。img_rgb: (..., 3), 末维是 RGB。

    注意空间一致性: 若调用点的张量已是当前 (gamma) 空间且原管线未线性化,
    保持一致先不加线性化, 作为最小改动。
    """
    M = bradford_correction_matrix(T_k, img_rgb.device, inverse=inverse)
    return img_rgb @ M.T


# --------------------------------------------------------------------------- #
# 模块 B: 全视角共享相机响应曲线
# --------------------------------------------------------------------------- #
class SharedCRF(nn.Module):
    """全视角共享的单调相机响应曲线。

    用 n 段正增量 (softplus 累积) 保证严格单调递增, 初始化≈线性。
    """

    def __init__(self, n: int = 256):
        super().__init__()
        self.n = n
        self.delta = nn.Parameter(torch.zeros(n))

    def curve(self) -> torch.Tensor:
        inc = torch.softplus(self.delta) + 1e-4   # 正增量 -> 单调
        c = torch.cumsum(inc, dim=0)
        c = c / c[-1].clamp_min(1e-8)             # 归一到 [0, 1]
        return c                                  # (n,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x in [0, 1], 任意 shape; 线性插值查表, 可微。"""
        c = self.curve()
        n = c.numel()
        xi = x.clamp(0, 1) * (n - 1)
        lo = xi.floor().long().clamp(0, n - 2)
        hi = lo + 1
        w = xi - lo.float()
        flat = c[lo.reshape(-1)] * (1 - w.reshape(-1)) + c[hi.reshape(-1)] * w.reshape(-1)
        return flat.reshape_as(x)


def apply_tone(img: torch.Tensor, e_k: torch.Tensor, crf: SharedCRF) -> torch.Tensor:
    """out = CRF(e_k · x)。e_k: per-view 正标量; crf: 共享单实例。"""
    return crf(e_k * img)


# --------------------------------------------------------------------------- #
# T_k / e_k 的范围映射 (用于参数生成器的新增输出头)
# --------------------------------------------------------------------------- #
def map_T(raw: torch.Tensor) -> torch.Tensor:
    """raw (任意实数) -> 色温 T_k ∈ [CCT_MIN, CCT_MAX], 区间对齐合成范围。"""
    return CCT_MIN + (CCT_MAX - CCT_MIN) * torch.sigmoid(raw)


def map_e(raw: torch.Tensor) -> torch.Tensor:
    """raw -> 正曝光增益 e_k, 初始≈1 (当 head 零初始化时 exp(0)=1)。"""
    return torch.exp(raw.clamp(-4.0, 4.0))


# --------------------------------------------------------------------------- #
# 合成退化管线: 干净图 -> 已知 (T_gt, K_gt) 的退化图
# --------------------------------------------------------------------------- #
def planckian_channel_gain(T_gt: torch.Tensor, device: torch.device) -> torch.Tensor:
    """色温 T_gt 下光源在 sRGB-linear 的逐通道相对增益 (绿通道归一)。

    = bradford_correction_matrix(T, inverse=True) 作用在中性灰上的结果,
      用作对干净图施加色偏的逐通道乘子。
    """
    neutral = torch.ones(3, device=device)
    cast = apply_color_adapt(neutral.reshape(1, 3), T_gt, inverse=True).reshape(3)
    cast = cast / cast[1].clamp_min(1e-6)             # 绿通道归一, 只留色偏
    return cast.clamp_min(1e-4)


def synthesize_degradation(
    img_clean: torch.Tensor, T_gt: torch.Tensor, K_gt: torch.Tensor
) -> torch.Tensor:
    """对干净图施加已知色温色偏 + 曝光增益, 返回退化图 (clamp 到 [0,1])。

    img_clean: (..., 3) in [0,1]; T_gt: 标量色温; K_gt: 标量曝光增益。
    退化模型 (与校正互为物理逆): img_deg = clamp(K_gt · gain(T_gt) ⊙ img_clean)。
    """
    gain = planckian_channel_gain(T_gt, img_clean.device)
    deg = K_gt * img_clean * gain
    return deg.clamp(0.0, 1.0)


def sample_degradation_params(
    generator: Optional[torch.Generator] = None,
    device: torch.device = "cpu",
    exp_name: str = "low",
):
    """为一个 view 采样真值退化参数。

    返回 (T_gt, K_gt) 标量 tensor。K_gt 区间按场景类型:
      low      -> 偏暗 (K<1)
      over_exp -> 偏亮 (K>1)
      其它      -> 跨曝光
    """
    def _u(a, b):
        return a + (b - a) * torch.rand(1, generator=generator, device=device)

    T_gt = _u(CCT_MIN, CCT_MAX).reshape(())
    if exp_name == "low":
        K_gt = _u(0.05, 0.4).reshape(())
    elif exp_name == "over_exp":
        K_gt = _u(1.5, 4.0).reshape(())
    else:
        K_gt = _u(0.2, 3.0).reshape(())
    return T_gt, K_gt


# --------------------------------------------------------------------------- #
# 可解释性日志
# --------------------------------------------------------------------------- #
class InterpretLogger:
    """逐 view 落盘预测值 (T_k, e_k) 与真值 (T_gt, K_gt) 到 csv。"""

    FIELDS = ["step", "image_id", "T_k", "e_k", "T_gt", "K_gt"]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def log(self, step, image_id, T_k=None, e_k=None, T_gt=None, K_gt=None):
        def _v(x):
            if x is None:
                return ""
            if torch.is_tensor(x):
                return float(x.detach().reshape(-1)[0].item())
            return float(x)

        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [int(step), int(image_id), _v(T_k), _v(e_k), _v(T_gt), _v(K_gt)]
            )
