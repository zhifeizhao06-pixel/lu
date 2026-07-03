import json
import math
import os
import time
from dataclasses import dataclass, field
from turtle import color
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import nerfview

from datasets.traj import generate_interpolated_path
import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    CrossAttention,
    CrossAttention_Curve,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
)

import matplotlib.pyplot as plt
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
gsplat_path = os.path.abspath(os.path.join(current_dir, '..', 'gsplat'))

sys.path.append(gsplat_path)

from rendering_double import rasterization_dual

from tools import pixel_project, pixel_project_back, LUT_mapping
from losses import L_spa, HistogramPriorLoss, gamma_curve, s_curve


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the dataset
    data_dir: str = "/data/umeiro0/users/cui/data/Multi-illudataset/Scene1/RGB_down4/GT"

    exp_name: str = ""   # Switch Conditions Here. overexposure: str = "over_exp"; varying exposure: str = "variance"
    method: str = ""   
    # Downsample factor for the dataset
    data_factor: int = 1    # data_factor 8 for Mip360 dataset
    # Directory to save results
    result_dir: str = "/data/umeiro0/users/cui/data/Multi-illudataset/Scene1/results"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps, max steps 10000 for LOM dataset training
    max_steps: int = 10_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [5_000, 7_000, 10_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [5_000, 7_000, 10_000])

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Low-light noise-aware optimization.  We use the common heteroscedastic
    # approximation Var[n] = alpha * intensity + beta for shot/read noise.
    noise_aware: bool = True
    noise_nll_lambda: float = 0.05
    noise_alpha_init: float = 0.01
    noise_beta_init: float = 0.001
    # Suppress densification when a Gaussian is observed mostly below the
    # estimated noise floor. Set to 0 to recover the original densification.
    confidence_densify: bool = True
    densify_confidence_min: float = 0.15
    densify_confidence_power: float = 1.0
    # Protect dark but structurally reliable edges. Image gradients are
    # normalized by the predicted noise of a pixel difference, sqrt(2)*sigma.
    structure_protection: bool = False
    edge_snr_low: float = 2.0
    edge_snr_high: float = 5.0
    structure_strength: float = 0.5
    # Accumulate image-plane gradients in a common world coordinate system.
    # Noise produces poorly aligned directions across views; real structure is
    # expected to generate a more coherent update direction.
    gradient_consensus: bool = False
    densify_consensus_min: float = 0.05
    densify_consensus_power: float = 0.5
    # Penalize needle-like Gaussians without penalizing thin, surface-aligned
    # discs. The ratio is s_max / s_mid, not s_max / s_min.
    needle_regularization: bool = True
    needle_reg_lambda: float = 5e-4
    needle_ratio_max: float = 5.0
    needle_reg_start: int = 1_000
    # Empirical Fisher information from the noise likelihood approximates how
    # observable each Gaussian's position and shape are under low-light noise.
    information_guidance: bool = False
    fisher_ema_decay: float = 0.95
    information_start: int = 1_000
    information_min_support: int = 3
    information_min: float = 0.01
    information_power: float = 0.25
    # Fisher can be used in two distinct ways. Direct densification weighting
    # is retained for ablation but disabled after it hurt perceptual quality.
    information_densify: bool = False
    # Recommended use: decouple geometry updates from appearance updates.
    information_gradient_gating: bool = False
    information_gate_floor: float = 0.5
    information_gate_power: float = 0.5

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    #refine_stop_iter: int = 15_000
    refine_stop_iter: int = 8_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Contrast Level
    constrast_level: float = 0.5

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)

cfg = tyro.cli(Config)

# if cfg.exp_name in ["low", "over_exp"]:
#     from datasets.colmap import Dataset, Parser
# else:
#     from datasets.colmap_mip360 import Dataset, Parser
if cfg.exp_name in ["low", "over_exp"]:
    from datasets.colmap import Dataset, Parser
elif cfg.exp_name == "variance":
    from datasets.colmap_mip360 import Dataset, Parser
else:
    from datasets.colmap import Dataset, Parser
    # from datasets.colmap_mip360_WB import Dataset, Parser


def create_splats_with_optimizers(
    points: Tensor,  # [N, 3]
    rgbs: Tensor,  # [N, 3]
    frame_nums: int, # Training Frame Number
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, torch.optim.Optimizer]:
    N = points.shape[0]

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]   point cloud position
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    params = [
        # name, value, lr
        ("means3d", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:    # Color is Here     
        # color is SH coefficients.     
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, 4**2, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)   # rgb to sh
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
        
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    # Eq.3 in our paper, a least-squares formula
    adjust_k = torch.nn.Parameter(torch.ones_like(colors[:, :1, :]), requires_grad=True)    # enhance, for multiply
    adjust_b = torch.nn.Parameter(torch.zeros_like(colors[:, :1, :]), requires_grad=True)   # bias, for add

    params.append(("adjust_k", adjust_k, 2.5e-3))
    params.append(("adjust_b", adjust_b, 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)    # parameter dict

    optimizers = [
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    ]
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.render_dir_depth = f"{cfg.result_dir}/renders_depth"
        os.makedirs(self.render_dir_depth, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            exp_name = cfg.exp_name,
            method = cfg.method,
            factor=cfg.data_factor, # down scale ratio
            normalize=True,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(    # Training Set
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val") # Validation Set
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        
        # Model
        feature_dim = 32 if cfg.app_opt else None
        # return GS-parameters & optimizers
        self.splats, self.optimizers = create_splats_with_optimizers(   # basic gaussian splatting
            torch.from_numpy(self.parser.points).float(),
            torch.from_numpy(self.parser.points_rgb / 255.0).float(),
            frame_nums = len(self.trainset),
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            init_opacity=cfg.init_opa,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))
        
        self.constrast_level = cfg.constrast_level
        
        curve = torch.linspace(0, 1, 255).unsqueeze(0).cuda()   # Luminance Curve
        self.curve = torch.nn.Parameter(curve)
        self.curve_optimizers = [
                torch.optim.Adam(
                    [self.curve],
                    # [self.curve, self.curve_2, self.curve_3],
                    lr=1e-3 * math.sqrt(cfg.batch_size),
                    weight_decay=1e-4,
                )
            ]
        
        
        self.curve_adjust = CrossAttention().to(self.device)    # Output the curve bias parameters, L_k_b
        self.curve_adjust_gamma = CrossAttention_Curve().to(self.device)    # Output the curve shape control parameters, Eq.9 

        self.adjust_optimizers = [
                torch.optim.Adam(
                    list(self.curve_adjust.parameters()) + list(self.curve_adjust_gamma.parameters()),
                    lr=1e-5 * math.sqrt(cfg.batch_size),
                    weight_decay=1e-5,
                )
            ]
        
        self.pesdo_curve = torch.nn.Parameter(torch.linspace(0, 1, 255).unsqueeze(0).cuda(), requires_grad=False)

        self.axis1_para = [torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]).cuda()) for _ in range(len(self.trainset))]
        self.axis2_para = [torch.nn.Parameter(torch.tensor([0.0, 0.0]).cuda()) for _ in range(len(self.trainset))]

        self.sat_optimizers = [
                torch.optim.Adam(
                    self.axis1_para + self.axis2_para,
                    lr=2e-4 * math.sqrt(cfg.batch_size),
                    weight_decay=1e-4,
                )
            ]

        # One shot-noise/read-noise pair per training view. Softplus keeps the
        # values positive; clamping in _noise_model prevents degenerate NLL
        # solutions during the first iterations.
        def inv_softplus(x: float) -> float:
            return math.log(math.expm1(x))

        noise_init = torch.tensor(
            [inv_softplus(cfg.noise_alpha_init), inv_softplus(cfg.noise_beta_init)],
            device=self.device,
        )
        self.noise_params = torch.nn.Parameter(
            noise_init[None].repeat(len(self.trainset), 1)
        )
        self.noise_optimizers = []
        if cfg.noise_aware:
            self.noise_optimizers = [
                torch.optim.Adam([self.noise_params], lr=2e-4, weight_decay=1e-6)
            ]
        
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
            "confidence": torch.zeros(n_gauss, device=self.device),
            "grad_world": torch.zeros(n_gauss, 3, device=self.device),
            "fisher_position": torch.zeros(n_gauss, device=self.device),
            "fisher_shape": torch.zeros(n_gauss, device=self.device),
            "view_support": torch.zeros(n_gauss, device=self.device),
        }

    def _noise_model(self, pixels: Tensor, image_ids: Tensor):
        """Return variance, signal confidence, and noise-normalized structure."""
        params = F.softplus(self.noise_params[image_ids.long()])
        alpha = params[:, 0].clamp(1e-5, 0.25)[:, None, None, None]
        beta = params[:, 1].clamp(1e-6, 0.05)[:, None, None, None]
        intensity = pixels.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        variance = (alpha * intensity + beta).clamp_min(1e-6)
        confidence = intensity / (intensity + variance.sqrt() + 1e-6)

        gray = intensity.permute(0, 3, 1, 2)
        grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
        gradient = gradient.permute(0, 2, 3, 1)

        # Independent neighboring samples have approximately twice the noise
        # variance in their difference. Values above edge_snr_high are treated
        # as reliable structure rather than sensor noise.
        edge_snr = gradient / (math.sqrt(2.0) * variance.sqrt() + 1e-6)
        snr_range = max(self.cfg.edge_snr_high - self.cfg.edge_snr_low, 1e-6)
        structure = ((edge_snr - self.cfg.edge_snr_low) / snr_range).clamp(0.0, 1.0)
        return variance, confidence.detach(), structure.detach()

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Learnable Parameters:
        means = self.splats["means3d"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,], sigmoid function

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:    
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
            
        else:  # Here 
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        adjust_k = self.splats["adjust_k"]  # 1090, 1, 3
        adjust_b = self.splats["adjust_b"]  # 1090, 1, 3
        
        colors_low = colors * adjust_k + adjust_b  # least squares: x_enh=a*x+b
        
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        
        render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info = rasterization_dual(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,    
            colors=colors,
            colors_low=colors_low,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,)
        
        return render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info   # return colors and alphas

    def train(self):
        cfg = self.cfg
        device = self.device

        loss_contrast = L_spa()     # spatial consistancy loss
        loss_histo = HistogramPriorLoss()   # curve control loss


        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        scheulers = [
            # means3d has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        # curve optimizer & curve adjustment optimizer & sat optimizer
        scheulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.curve_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.noise_aware:
            scheulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.noise_optimizers[0], gamma=0.1 ** (1.0 / max_steps)
                )
            )
        scheulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.adjust_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        scheulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.sat_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            scheulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders_enh, renders_low, alphas_enh, alphas_low, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB",
            )
            if renders_low.shape[-1] == 4:
                colors_low, depths_low = renders_low[..., 0:3], renders_low[..., 3:4]
                colors_enh, depths_enh = renders_enh[..., 0:3], renders_enh[..., 3:4]
            else:
                colors_low, depths_low = renders_low, None
                colors_enh, depths_enh = renders_enh, None

            if cfg.random_bkgd: # False
                bkgd = torch.rand(1, 3, device=device)
                colors_low = colors_low + bkgd * (1.0 - alphas_low)
                colors_enh = colors_enh + bkgd * (1.0 - alphas_enh)

            info["means2d"].retain_grad()  # used for running stats
            
            curve_adj_bias = self.curve_adjust(pixels.permute(0,3,1,2), camtoworlds) # encode low-light GT and camera position to get adjust curve
            
            gamma_alpha_beta = self.curve_adjust_gamma(pixels.permute(0,3,1,2), camtoworlds)

            curve_adj = torch.clamp(self.curve + curve_adj_bias, 0, 1)    # Clamp the curve in range of (0, 1)

            normal= (self.axis1_para[image_ids] + torch.Tensor([1, 0, 0]).to(colors_low.device)).unsqueeze(0)
            
            normal2 = (self.axis2_para[image_ids] + torch.Tensor([1, 0]).to(colors_low.device)).unsqueeze(0)
            
            bias = torch.zeros([1, 3]).to(colors_low.device)
            
            t1s, t2s, t3s, bias  = pixel_project(pixels.permute(0,3,1,2), normal, normal2, bias)
            t1s_out = [LUT_mapping(t1s, curve_adj), t1s[1], t1s[2], t1s[3]] 
            t2s_out = [LUT_mapping(t2s, curve_adj), t2s[1], t2s[2], t2s[3]] 
            t3s_out = [LUT_mapping(t3s, curve_adj), t3s[1], t3s[2], t3s[3]] 

            pixels_enh = pixel_project_back(t1s_out, t2s_out, t3s_out, bias).permute(0,2,3,1)
            
            gamma = gamma_alpha_beta[:,0]
            alpha, beta = gamma_alpha_beta[:,1], gamma_alpha_beta[:,2]
            
            gamma = torch.Tensor([1.0]).to(device) + 0.1*gamma
            alpha = torch.Tensor([0.5]).to(device) + 0.002*alpha 
            beta = torch.Tensor([1.0]).to(device) + 0.002*beta
            
            pesdo_curve = gamma_curve(self.pesdo_curve, gamma)  # Pseudo-gamma curve
            pesdo_curve = s_curve(pesdo_curve, alpha, beta) # Pseudo-scurve curve
            

            con_degree = (self.constrast_level/torch.mean(pixels)).item()   # frame-adaptive contrast degree, Eq.8 in paper
            loss_co = loss_contrast(pixels.permute(0,3,1,2), colors_enh.permute(0,3,1,2), contrast=con_degree)
            
            l1loss = F.l1_loss(colors_low, pixels)
            ssimloss = 1.0 - self.ssim(pixels.permute(0,3,1,2), colors_low.permute(0,3,1,2))
            loss_regress_low = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            noise_nll = torch.zeros((), device=device)
            signal_confidence = None
            structure_confidence = None
            densify_confidence = None
            if cfg.noise_aware:
                (
                    noise_variance,
                    signal_confidence,
                    structure_confidence,
                ) = self._noise_model(pixels, image_ids)
                densify_confidence = signal_confidence
                if cfg.structure_protection:
                    densify_confidence = signal_confidence + cfg.structure_strength * (
                        1.0 - signal_confidence
                    ) * structure_confidence
                residual2 = (colors_low - pixels).square().mean(dim=-1, keepdim=True)
                noise_nll = 0.5 * (
                    residual2 / noise_variance + torch.log(noise_variance)
                ).mean()

            needle_loss = torch.zeros((), device=device)
            if cfg.needle_regularization and step >= cfg.needle_reg_start:
                # Work in log-scale space for stable gradients. Sorting is
                # piecewise differentiable and routes gradients to the two
                # axes that form the needle ratio.
                sorted_log_scales = self.splats["scales"].sort(dim=-1).values
                log_needle_ratio = sorted_log_scales[:, 2] - sorted_log_scales[:, 1]
                excess = F.relu(
                    log_needle_ratio - math.log(cfg.needle_ratio_max)
                )
                # Visible/opaque needles matter more, while a small floor also
                # prevents nearly transparent outliers from growing unchecked.
                opacity_weight = 0.1 + 0.9 * torch.sigmoid(
                    self.splats["opacities"].detach()
                )
                needle_loss = (opacity_weight * excess.square()).mean()

            l1loss_enh = F.l1_loss(colors_enh, pixels_enh)  # enhancement loss constrain
            ssimloss_enh = 1.0 - self.ssim(pixels_enh.permute(0,3,1,2), colors_enh.permute(0,3,1,2))
            loss_regress_enh = l1loss_enh * (1.0 - cfg.ssim_lambda) + ssimloss_enh * cfg.ssim_lambda
            
            hist_loss = loss_histo(curve_adj, pixels, pesdo_curve, step, exp_name=cfg.exp_name)
            
            loss = (
                loss_regress_low
                + 0.5 * loss_regress_enh
                + loss_co
                + 10 * hist_loss
                + cfg.noise_nll_lambda * noise_nll
                + cfg.needle_reg_lambda * needle_loss
            )

            information_grads = None
            if (
                cfg.information_guidance
                and cfg.noise_aware
                and step >= cfg.information_start
            ):
                # The squared score of the heteroscedastic likelihood is an
                # empirical diagonal-Fisher approximation. Across randomly
                # sampled views it measures whether geometry is consistently
                # observable above the estimated sensor noise.
                information_grads = torch.autograd.grad(
                    noise_nll,
                    (
                        self.splats["means3d"],
                        self.splats["scales"],
                        self.splats["quats"],
                    ),
                    retain_graph=True,
                    allow_unused=True,
                )
                # The rasterizer exposes retained screen-space gradients for
                # densification. Although autograd.grad does not accumulate
                # into the requested leaf parameters, backward hooks of the
                # custom rasterizer can populate means2d.grad/absgrad. Clear
                # those auxiliary values so the following full backward pass
                # is the sole source used by update_running_stats().
                info["means2d"].grad = None
                if hasattr(info["means2d"], "absgrad"):
                    info["means2d"].absgrad = None

            loss.backward()

            if information_grads is not None:
                self.update_fisher_stats(info, information_grads)

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                if cfg.noise_aware:
                    self.writer.add_scalar("train/noise_nll", noise_nll.item(), step)
                    noise_values = F.softplus(self.noise_params[image_ids.long()]).mean(0)
                    self.writer.add_scalar("train/noise_alpha", noise_values[0].item(), step)
                    self.writer.add_scalar("train/noise_beta", noise_values[1].item(), step)
                    self.writer.add_scalar(
                        "train/signal_confidence",
                        signal_confidence.mean().item(),
                        step,
                    )
                    self.writer.add_scalar(
                        "train/structure_confidence",
                        structure_confidence.mean().item(),
                        step,
                    )
                    self.writer.add_scalar(
                        "train/densify_confidence",
                        densify_confidence.mean().item(),
                        step,
                    )
                if cfg.needle_regularization:
                    self.writer.add_scalar("train/needle_loss", needle_loss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means3d"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                
                if cfg.tb_save_image:
                    canvas = torch.cat([colors_enh, pixels_enh], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)

                    canvas_low = torch.cat([colors_low, pixels], dim=2).detach().cpu().numpy()
                    canvas_low = canvas_low.reshape(-1, *canvas_low.shape[2:])
                    self.writer.add_image("train/render_low", canvas_low, step)

                self.writer.flush()

            # update running stats for prunning & growing
            if step < cfg.refine_stop_iter:
                self.update_running_stats(
                    info, densify_confidence, camtoworlds.detach(), Ks.detach()
                )

                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    mean_confidence = self.running_stats["confidence"] / self.running_stats[
                        "count"
                    ].clamp_min(1)
                    if cfg.confidence_densify and cfg.noise_aware:
                        grads = grads * mean_confidence.pow(cfg.densify_confidence_power)

                    consensus = self.running_stats["grad_world"].norm(dim=-1) / (
                        self.running_stats["confidence"].clamp_min(1e-6)
                    )
                    consensus = consensus.clamp(0.0, 1.0)
                    if cfg.gradient_consensus and cfg.noise_aware:
                        grads = grads * consensus.pow(cfg.densify_consensus_power)

                    information_confidence = self.get_information_confidence()
                    if (
                        cfg.information_guidance
                        and cfg.information_densify
                        and cfg.noise_aware
                    ):
                        grads = grads * information_confidence.pow(
                            cfg.information_power
                        )

                    # grow GSs
                    is_grad_high = grads >= cfg.grow_grad2d
                    if cfg.confidence_densify and cfg.noise_aware:
                        is_grad_high = is_grad_high & (
                            mean_confidence >= cfg.densify_confidence_min
                        )
                    if cfg.gradient_consensus and cfg.noise_aware:
                        is_grad_high = is_grad_high & (
                            consensus >= cfg.densify_consensus_min
                        )
                    if (
                        cfg.information_guidance
                        and cfg.information_densify
                        and cfg.noise_aware
                    ):
                        has_support = self.running_stats["view_support"] >= (
                            cfg.information_min_support
                        )
                        is_grad_high = is_grad_high & (
                            (~has_support)
                            | (information_confidence >= cfg.information_min)
                        )
                    is_small = (
                        torch.exp(self.splats["scales"]).max(dim=-1).values
                        <= cfg.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < cfg.prune_opa
                    if step > cfg.reset_every:
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # reset running stats
                    self.running_stats["grad2d"].zero_()
                    self.running_stats["count"].zero_()
                    self.running_stats["confidence"].zero_()
                    self.running_stats["grad_world"].zero_()

                if step % cfg.reset_every == 0:
                    self.reset_opa(cfg.prune_opa * 2.0)

            # Parameter-update decoupling: geometry with weak Fisher support
            # receives conservative updates, while SH/color and exposure
            # parameters remain free to explain appearance changes.
            if (
                cfg.information_guidance
                and cfg.information_gradient_gating
                and cfg.noise_aware
                and step >= cfg.information_start
            ):
                position_confidence, shape_confidence = (
                    self.get_information_confidence_components()
                )
                position_gate = cfg.information_gate_floor + (
                    1.0 - cfg.information_gate_floor
                ) * position_confidence.pow(cfg.information_gate_power)
                shape_gate = cfg.information_gate_floor + (
                    1.0 - cfg.information_gate_floor
                ) * shape_confidence.pow(cfg.information_gate_power)

                geometry_gates = {
                    "means3d": position_gate,
                    "scales": shape_gate,
                    "quats": shape_gate,
                }
                for name, gate in geometry_gates.items():
                    grad = self.splats[name].grad
                    if grad is None or grad.is_sparse:
                        continue
                    gate_shape = [len(gate)] + [1] * (grad.dim() - 1)
                    grad.mul_(gate.view(gate_shape))

                if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                    self.writer.add_scalar(
                        "information/position_gate_mean", position_gate.mean().item(), step
                    )
                    self.writer.add_scalar(
                        "information/shape_gate_mean", shape_gate.mean().item(), step
                    )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.curve_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.adjust_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.sat_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.noise_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in scheulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                scales = torch.exp(self.splats["scales"].detach())
                sorted_scales = scales.sort(dim=-1).values
                elongation = sorted_scales[:, 2] / sorted_scales[:, 0].clamp_min(1e-8)
                # max/mid detects needle-like Gaussians, while mid/min detects
                # thin surface-aligned discs. max/min alone conflates the two.
                needle_ratio = sorted_scales[:, 2] / sorted_scales[:, 1].clamp_min(1e-8)
                flat_ratio = sorted_scales[:, 1] / sorted_scales[:, 0].clamp_min(1e-8)
                opacity = torch.sigmoid(self.splats["opacities"].detach())
                opaque = opacity > 0.1
                opaque_count = opaque.sum().clamp_min(1)
                stats.update(
                    {
                        "elongation_mean": elongation.mean().item(),
                        "elongation_median": elongation.median().item(),
                        "elongation_gt5": (elongation > 5).float().mean().item(),
                        "elongation_gt10": (elongation > 10).float().mean().item(),
                        "elongation_gt20": (elongation > 20).float().mean().item(),
                        "needle_mean": needle_ratio.mean().item(),
                        "needle_median": needle_ratio.median().item(),
                        "needle_gt5": (needle_ratio > 5).float().mean().item(),
                        "needle_gt10": (needle_ratio > 10).float().mean().item(),
                        "flat_mean": flat_ratio.mean().item(),
                        "flat_median": flat_ratio.median().item(),
                        "flat_gt10": (flat_ratio > 10).float().mean().item(),
                        "opaque_fraction": opaque.float().mean().item(),
                        "opaque_needle_gt5": (
                            ((needle_ratio > 5) & opaque).sum() / opaque_count
                        ).item(),
                        "opaque_needle_gt10": (
                            ((needle_ratio > 10) & opaque).sum() / opaque_count
                        ).item(),
                        "opacity_weighted_needle": (
                            (needle_ratio * opacity).sum() / opacity.sum().clamp_min(1e-8)
                        ).item(),
                    }
                )
                if cfg.information_guidance:
                    info_confidence = self.get_information_confidence()
                    position_confidence, shape_confidence = (
                        self.get_information_confidence_components()
                    )
                    supported = self.running_stats["view_support"] >= (
                        cfg.information_min_support
                    )
                    if supported.any():
                        supported_confidence = info_confidence[supported]
                        stats.update(
                            {
                                "information_supported_fraction": supported.float()
                                .mean()
                                .item(),
                                "information_mean": supported_confidence.mean().item(),
                                "information_median": supported_confidence.median().item(),
                                "information_below_min": (
                                    supported_confidence < cfg.information_min
                                )
                                .float()
                                .mean()
                                .item(),
                                "position_gate_mean": (
                                    cfg.information_gate_floor
                                    + (1.0 - cfg.information_gate_floor)
                                    * position_confidence[supported].pow(
                                        cfg.information_gate_power
                                    )
                                )
                                .mean()
                                .item(),
                                "shape_gate_mean": (
                                    cfg.information_gate_floor
                                    + (1.0 - cfg.information_gate_floor)
                                    * shape_confidence[supported].pow(
                                        cfg.information_gate_power
                                    )
                                )
                                .mean()
                                .item(),
                            }
                        )
                for key in (
                    "elongation_mean",
                    "elongation_gt5",
                    "elongation_gt10",
                    "elongation_gt20",
                    "needle_mean",
                    "needle_gt5",
                    "needle_gt10",
                    "flat_mean",
                    "flat_gt10",
                    "opaque_needle_gt5",
                    "opaque_needle_gt10",
                    "opacity_weighted_needle",
                ):
                    self.writer.add_scalar(f"geometry/{key}", stats[key], step)
                for key in (
                    "information_supported_fraction",
                    "information_mean",
                    "information_median",
                    "information_below_min",
                    "position_gate_mean",
                    "shape_gate_mean",
                ):
                    if key in stats:
                        self.writer.add_scalar(f"information/{key}", stats[key], step)
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                        "curve": self.curve.detach(),
                        "curve_adjust": self.curve_adjust.state_dict(),
                        "curve_adjust_gamma": self.curve_adjust_gamma.state_dict(),
                        "axis1_para": [p.detach() for p in self.axis1_para],
                        "axis2_para": [p.detach() for p in self.axis2_para],
                        "noise_params": self.noise_params.detach(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def update_fisher_stats(self, info: Dict, information_grads):
        """Update per-Gaussian empirical Fisher information for geometry."""
        if self.cfg.packed:
            visible_ids = info["gaussian_ids"].unique()
            visible = torch.zeros(
                len(self.splats["means3d"]), device=self.device, dtype=torch.bool
            )
            visible[visible_ids] = True
        else:
            visible = (info["radii"] > 0).any(dim=0)

        def squared_score(grad: Optional[Tensor]) -> Tensor:
            if grad is None:
                return torch.zeros(len(visible), device=self.device)
            if grad.is_sparse:
                grad = grad.to_dense()
            return grad.detach().float().square().flatten(1).mean(dim=-1)

        position_score = squared_score(information_grads[0])
        shape_score = squared_score(information_grads[1]) + squared_score(
            information_grads[2]
        )
        decay = self.cfg.fisher_ema_decay
        self.running_stats["fisher_position"][visible] = (
            decay * self.running_stats["fisher_position"][visible]
            + (1.0 - decay) * position_score[visible]
        )
        self.running_stats["fisher_shape"][visible] = (
            decay * self.running_stats["fisher_shape"][visible]
            + (1.0 - decay) * shape_score[visible]
        )
        self.running_stats["view_support"][visible] += 1.0

    @torch.no_grad()
    def get_information_confidence_components(self):
        """Return separately normalized position and shape information."""
        support = self.running_stats["view_support"] >= (
            self.cfg.information_min_support
        )

        def normalize(values: Tensor) -> Tensor:
            positive = support & torch.isfinite(values) & (values > 0)
            confidence = torch.ones_like(values)
            if positive.any():
                reference = values[positive].median().clamp_min(1e-20)
                confidence[support] = values[support] / (
                    values[support] + reference
                )
            return confidence.clamp(0.0, 1.0)

        position_confidence = normalize(self.running_stats["fisher_position"])
        shape_confidence = normalize(self.running_stats["fisher_shape"])
        return position_confidence, shape_confidence

    @torch.no_grad()
    def get_information_confidence(self) -> Tensor:
        """Combine position/shape information into a robust [0, 1] score."""
        position_confidence, shape_confidence = (
            self.get_information_confidence_components()
        )
        return torch.sqrt(position_confidence * shape_confidence)

    @torch.no_grad()
    def update_running_stats(
        self,
        info: Dict,
        confidence_map: Optional[Tensor] = None,
        camtoworlds: Optional[Tensor] = None,
        Ks: Optional[Tensor] = None,
    ):
        """Update running stats."""
        cfg = self.cfg

        # normalize grads to [-1, 1] screen space
        if cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size

        def lift_to_world(
            image_grads: Tensor, camera_ids: Tensor, confidence: Tensor
        ) -> Tensor:
            """Approximately lift 2D gradient directions into world space."""
            if camtoworlds is None or Ks is None:
                return torch.zeros(
                    len(image_grads), 3, device=image_grads.device, dtype=image_grads.dtype
                )
            camera_ids = camera_ids.long()
            fx = Ks[camera_ids, 0, 0].clamp_min(1e-6)
            fy = Ks[camera_ids, 1, 1].clamp_min(1e-6)
            right = camtoworlds[camera_ids, :3, 0]
            up = camtoworlds[camera_ids, :3, 1]
            world = (
                image_grads[:, 0:1] / fx[:, None] * right
                + image_grads[:, 1:2] / fy[:, None] * up
            )
            world = F.normalize(world, dim=-1, eps=1e-8)
            return world * confidence[:, None]

        if cfg.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids, dtype=torch.int)
            )
            if confidence_map is not None:
                camera_ids = info["camera_ids"].long()
                xy = info["means2d"].detach().round().long()
                xs = xy[:, 0].clamp(0, info["width"] - 1)
                ys = xy[:, 1].clamp(0, info["height"] - 1)
                conf = confidence_map[camera_ids, ys, xs, 0]
                self.running_stats["confidence"].index_add_(0, gs_ids, conf)
                world_grads = lift_to_world(grads, camera_ids, conf)
                self.running_stats["grad_world"].index_add_(0, gs_ids, world_grads)
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )
            if confidence_map is not None:
                camera_ids, _ = torch.where(sel)
                xy = info["means2d"].detach()[sel].round().long()
                xs = xy[:, 0].clamp(0, info["width"] - 1)
                ys = xy[:, 1].clamp(0, info["height"] - 1)
                conf = confidence_map[camera_ids, ys, xs, 0]
                self.running_stats["confidence"].index_add_(0, gs_ids, conf)
                world_grads = lift_to_world(grads[sel], camera_ids, conf)
                self.running_stats["grad_world"].index_add_(0, gs_ids, world_grads)

    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            torch.cuda.synchronize()
            tic = time.time()
            colors_enh, colors_low, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            
            depth_low = colors_low[:, :, :, 3:]
            depth_enh = colors_enh[:, :, :, 3:]
            colors_low = colors_low[:, :, :, :3]
            
            colors_enh = torch.clamp(colors_enh[:, :, :, :3], 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            
            canvas = torch.cat([colors_low, colors_enh], dim=2).squeeze(0).cpu().numpy()
            
            imageio.imwrite(
                f"{self.render_dir_depth}/val_{i:04d}_depth_low.png", (depth_low.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            )

            imageio.imwrite(
                f"{self.render_dir_depth}/val_{i:04d}_depth_enh.png", (depth_enh.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_low.png", (colors_low.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_enh.png", (colors_enh.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_enh = colors_enh.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors_enh, pixels))
            metrics["ssim"].append(self.ssim(colors_enh, pixels))
            metrics["lpips"].append(self.lpips(colors_enh, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        # print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        camtoworlds = self.parser.camtoworlds[10:60]
        camtoworlds = generate_interpolated_path(camtoworlds, 10)  # [N, 3, 4]
        
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]
        
        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders_enh, renders_low, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders_enh[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders_enh[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            canvas = colors
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=50)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors_enh, render_colors_low, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors_enh[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        # New checkpoints contain every trainable low-light component. Keep
        # loading backward compatible with the original public checkpoints.
        if "curve" in ckpt:
            runner.curve.data.copy_(ckpt["curve"])
        if "curve_adjust" in ckpt:
            runner.curve_adjust.load_state_dict(ckpt["curve_adjust"])
        if "curve_adjust_gamma" in ckpt:
            runner.curve_adjust_gamma.load_state_dict(ckpt["curve_adjust_gamma"])
        if "axis1_para" in ckpt:
            for dst, src in zip(runner.axis1_para, ckpt["axis1_para"]):
                dst.data.copy_(src)
        if "axis2_para" in ckpt:
            for dst, src in zip(runner.axis2_para, ckpt["axis2_para"]):
                dst.data.copy_(src)
        if (
            "noise_params" in ckpt
            and runner.noise_params.shape == ckpt["noise_params"].shape
        ):
            runner.noise_params.data.copy_(ckpt["noise_params"])
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
