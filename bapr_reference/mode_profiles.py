# Mode profiles for BA-PR / CS-BAPR SAC environment
# Each mode defines a distinct environmental regime with different parameters
#
# ===== CS-BAPR 扩展说明 =====
# 模式分为两类：
#   TRAIN_MODES  — 训练时可随机切换到的模式（OD 范围 0.3x–5x）
#   OOD_MODES    — 仅测试时注入，训练中从未见过（OD 10x–50x+）
#
# CS-BAPR 的核心验证场景：
#   平时客流 3–5 人/站/分钟 → 大型活动突变到 100–150 人/站/分钟
#   即 OD 倍率 20x–50x，远超训练分布，标准 NN 无法处理
# ================================================================

# ─────────────────────────────────────────────────
# 训练模式：训练时随机切换（BA-PR 原有 5 种）
# OD 范围: 0.3x – 5x（训练分布内）
# ─────────────────────────────────────────────────
TRAIN_MODES = {
    # ─── 正常模式 ───
    "normal": {
        "speed_mean_scale": 1.0,      # 速度均值不变
        "sigma": 1.5,                  # 标准方差
        "speed_cap": 15,               # 正常限速
        "od_global_mult": 1.0,         # 标准客流
        "station_od_overrides": {},     # 无站点特殊干预
        "affected_routes": None,       # 所有路段
    },

    # ─── 严重拥堵：某段道路事故/施工 ───
    "congestion_severe": {
        "speed_mean_scale": 0.3,       # 均值降到 30%
        "sigma": 3.0,                  # 波动也变大
        "speed_cap": 5,                # 限速降到 5 m/s
        "od_global_mult": 1.0,
        "station_od_overrides": {},
        "affected_routes": [3, 4, 5, 6],  # 只影响路段 3-6（中段路网）
    },

    # ─── 客流激增：学校放学 / 大型活动 ───
    "demand_surge": {
        "speed_mean_scale": 1.0,
        "sigma": 1.5,
        "speed_cap": 15,
        "od_global_mult": 1.5,         # 全局 1.5 倍
        "station_od_overrides": {      # 特定站点 OD 暴涨
            "X05": 5.0,                # 站点 X05 客流 5 倍
            "X06": 4.0,
            "X07": 3.0,
        },
        "affected_routes": None,
    },

    # ─── 全线瘫痪：极端天气 ───
    "extreme_weather": {
        "speed_mean_scale": 0.4,       # 全线均值降 60%
        "sigma": 4.0,                  # 极大方差
        "speed_cap": 8,                # 全线限速
        "od_global_mult": 0.3,         # 乘客也减少
        "station_od_overrides": {},
        "affected_routes": None,
    },

    # ─── 局部路段封闭 + 周边客流转移 ───
    "partial_closure": {
        "speed_mean_scale": 0.15,      # 封闭路段速度降到 15%
        "sigma": 1.0,
        "speed_cap": 3,                # 封闭路段限速 3
        "od_global_mult": 1.0,
        "station_od_overrides": {
            "X08": 4.0,
            "X09": 4.0,
            "X10": 3.0,
        },
        "affected_routes": [7, 8, 9],
    },
}

# ─────────────────────────────────────────────────
# OOD 测试模式：仅在评估时注入，训练中从未出现
# OD 范围: 10x – 50x+（远超训练分布）
#
# 对应 CS-BAPR OOD 界（Proposition 4.15, 无 Assumption 4）：
#   ‖π(x_ood) - f_real(x_ood)‖ ≤ δ + ε·‖d‖ + (L_pol + M)·‖d‖²
#   其中:
#     δ = 训练域 base accuracy
#     ε = Jacobian Consistency error（训练损失可控）
#     L_pol = NAU/NMU 架构的导数 Lipschitz 常数
#     M = 真实物理动力学的二阶光滑度
#     ‖d‖ 正比于 OD 倍率偏离训练分布的程度
#
#   Theorem 4.18（fencing theorem 版本）给出更紧的 L/2 系数：
#     ‖π(x_ood) - f_real(x_ood)‖ ≤ δ + ε·‖d‖ + L/2·‖d‖²
# ─────────────────────────────────────────────────
OOD_MODES = {
    # ─── 大型活动突变（核心场景）───
    # 演唱会/体育赛事/跨年，关键站点客流从 3–5 → 100–150
    # OD 倍率 30x–50x，道路也因散场人流变慢
    "mega_event": {
        "speed_mean_scale": 0.5,       # 道路因大量人流变慢到 50%
        "sigma": 2.0,                  # 中等波动
        "speed_cap": 10,               # 人流密集区限速
        "od_global_mult": 3.0,         # 全局 3x（周边站也有溢出）
        "station_od_overrides": {
            "X05": 30.0,               # 活动场馆站: 30x → 客流 3→90
            "X06": 50.0,               # 主入口站:   50x → 客流 3→150 ✓
            "X07": 20.0,               # 次入口站:   20x → 客流 3→60
            "X04": 10.0,               # 辐射站 1:   10x
            "X08": 10.0,               # 辐射站 2:   10x
        },
        "affected_routes": None,
        # CS-BAPR 元数据
        "_ood_severity": "extreme",    # 严重程度标签
        "_description": "大型活动(演唱会/赛事): 关键站客流 30-50x",
        "_expected_od_range": [10, 50], # 期望 OD 倍率范围
    },

    # ─── 节假日全线爆满 ───
    # 春运/黄金周/元旦，全线均匀暴涨 10–15x
    # 与 mega_event 的区别：均匀分布 vs 局部集中
    "holiday_rush": {
        "speed_mean_scale": 0.7,       # 轻度拥堵
        "sigma": 2.5,                  # 较大波动
        "speed_cap": 12,               # 全线轻微限速
        "od_global_mult": 10.0,        # 全线 10x（核心差异）
        "station_od_overrides": {
            "X03": 15.0,               # 商圈站: 15x
            "X05": 15.0,               # 交通枢纽: 15x
            "X10": 12.0,               # 住宅区: 12x
        },
        "affected_routes": None,
        "_ood_severity": "high",
        "_description": "节假日全线: 均匀 10x + 热点 15x",
        "_expected_od_range": [10, 15],
    },

    # ─── 突发灾害疏散 ───
    # 地震预警/洪水/火灾疏散，极端单向客流 + 道路接近瘫痪
    # 测试速度+客流双极端同时出现的鲁棒性
    "emergency_evacuation": {
        "speed_mean_scale": 0.2,       # 道路接近瘫痪
        "sigma": 5.0,                  # 极端波动（有的路完全堵死）
        "speed_cap": 5,                # 严格限速
        "od_global_mult": 8.0,         # 全线 8x
        "station_od_overrides": {
            "X06": 40.0,               # 疏散核心站: 40x
            "X07": 35.0,               # 疏散次站:   35x
            "X05": 25.0,               # 周边站:     25x
        },
        "affected_routes": None,
        "_ood_severity": "extreme",
        "_description": "突发灾害疏散: 速度20% + 客流 25-40x",
        "_expected_od_range": [8, 40],
    },

    # ─── 参数化 OOD 扫描模式 ───
    # 用于画 "实际误差 vs OOD 理论界" 曲线
    # od_global_mult 由外部在运行时通过 set_ood_multiplier() 设置
    # 默认值 1.0 只是占位，实际使用时会被覆盖
    "ood_parametric": {
        "speed_mean_scale": 1.0,       # 速度不变（隔离客流变量）
        "sigma": 1.5,                  # 标准波动
        "speed_cap": 15,               # 正常限速
        "od_global_mult": 1.0,         # ← 运行时被 set_ood_multiplier() 覆盖
        "station_od_overrides": {},     # ← 运行时可选覆盖
        "affected_routes": None,
        "_ood_severity": "parametric",
        "_description": "参数化扫描: 仅调 OD 倍率，用于画 OOD 界曲线",
        "_expected_od_range": [1, 100],
    },

    # ─── 速度+客流双极端扫描 ───
    # 同时调速度和客流，验证多维 OOD
    "ood_dual_extreme": {
        "speed_mean_scale": 0.2,       # 速度骤降到 20%
        "sigma": 4.0,                  # 高波动
        "speed_cap": 5,                # 严格限速
        "od_global_mult": 20.0,        # 全线 20x
        "station_od_overrides": {
            "X05": 40.0,               # 核心站 40x
            "X06": 40.0,
        },
        "affected_routes": None,
        "_ood_severity": "extreme",
        "_description": "速度+客流双极端: 速度20% + 全线20x/核心40x",
        "_expected_od_range": [20, 40],
    },
}

# ─────────────────────────────────────────────────
# 向后兼容：MODE_PROFILES = TRAIN_MODES（原有代码不受影响）
# ─────────────────────────────────────────────────
MODE_PROFILES = TRAIN_MODES

# ─────────────────────────────────────────────────
# 全部模式（训练+OOD），用于需要完整列表的场景
# ─────────────────────────────────────────────────
ALL_MODES = {**TRAIN_MODES, **OOD_MODES}

# ─────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────
def make_parametric_ood(od_mult, speed_scale=1.0, sigma=1.5, speed_cap=15,
                        station_overrides=None):
    """
    创建参数化 OOD 模式配置，用于扫描实验。

    Args:
        od_mult:           全局 OD 倍率 (e.g., 20.0 = 20x 正常客流)
        speed_scale:       速度均值缩放 (0-1, default 1.0 = 不变)
        sigma:             速度 lognormal sigma (default 1.5)
        speed_cap:         动态速度上限 (default 15)
        station_overrides: {station_name: od_mult} 站点级覆盖

    Returns:
        dict: 可直接传入 env._apply_mode() 使用的模式配置

    Example:
        # 扫描 OD 从 1x 到 100x
        for mult in [1, 5, 10, 20, 50, 100]:
            profile = make_parametric_ood(mult)
            env._apply_mode_profile(profile)
    """
    return {
        "speed_mean_scale": speed_scale,
        "sigma": sigma,
        "speed_cap": speed_cap,
        "od_global_mult": float(od_mult),
        "station_od_overrides": station_overrides or {},
        "affected_routes": None,
        "_ood_severity": "parametric",
        "_description": f"parametric OOD sweep: od={od_mult}x, speed={speed_scale}",
        "_expected_od_range": [od_mult, od_mult],
    }


def get_ood_sweep_configs(od_range=None, include_speed_variation=False):
    """
    生成用于 OOD 参数扫描的配置列表。

    Args:
        od_range:   OD 倍率列表, 默认 [1, 2, 5, 10, 20, 30, 50, 75, 100]
        include_speed_variation: 是否同时扫速度变化

    Returns:
        list of (label, profile_dict) tuples

    Example:
        for label, profile in get_ood_sweep_configs():
            env._apply_mode_profile(profile)
            # ... run episode, collect metrics
            results[label] = metrics
    """
    if od_range is None:
        od_range = [1, 2, 5, 10, 20, 30, 50, 75, 100]

    configs = []
    for mult in od_range:
        if include_speed_variation:
            for speed in [1.0, 0.5, 0.2]:
                label = f"od{mult}x_spd{int(speed*100)}pct"
                configs.append((label, make_parametric_ood(mult, speed_scale=speed)))
        else:
            label = f"od{mult}x"
            configs.append((label, make_parametric_ood(mult)))
    return configs
