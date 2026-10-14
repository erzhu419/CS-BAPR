# CS-BAPR 全盘问题与研究交接

更新时间：2026-10-14，Asia/Shanghai。仓库：`erzhu419/CS-BAPR`，
分支：`codex/general-ood-rl`。最新实验为 v150，四组正式任务已完成并核验。

## 给 GPT6 的请求

请基于以下代码、协议和结果，判断这个研究现在真正成立的贡献是什么、
为什么负载调节任务有效而通用 Hopper 不稳定，并设计最小的一轮判别实验。
可以删除 NAU、SINDy 或重定义方法，但需要说明保留机制的必要性。
请优先决定研究对象和因果问题，再决定改算法还是改环境；不要仅罗列
context inference、meta-learning、VAE、ensemble 等候选组件。
用户已经理解 RE-SAC、ESCP、meta-learning、VAE 和 BAPR。

## 1. 当前结论

已有真实的局部正结果，但尚没有一个可靠复现公交式优势的通用极端 OOD 版本。
同名 CS-BAPR 已经包含不同方法路线，需要先厘清身份。

| 路线 | 已成立的结果 | 当前研究问题 |
|---|---|---|
| 公交历史主稿 | 稳定化配方改善所比较的 BAPR；架构排序随需求场景变化 | 主结果未启用 SINDy/IRM，不能作为符号因果机制证据；历史原始逐种子材料缺失 |
| 非线性负载 v10--v13 | 去 NAU、保留 SINDy 参考和 actor Jacobian 约束，优于所比较的学习器；独立 seed 和新扰动路径验证 | 大部分收益来自拟合反馈；相对直接参考的额外 RL 收益小，扩大状态范围后不稳定 |
| 持续负载 v31--v36 | Jacobian 约束开关有较大收益；改参考也有局部 OOD 收益 | 不稳定超过直接参考；还没有完整强基线确认 |
| Hopper belief/context 路线 | 有正常策略、信号识别和个别恢复正结果 | 识别不等于适应，训练漂移混入信号；极端 OOD 尚不可靠 |
| v144--v150 标准 SAC 校准 | 正常环境可学；部分短重力冲击可恢复；前冲击噪声影响后续鲁棒性 | 这些不是 CS-BAPR、RE-SAC 或 BAPR 算法对比 |

实验编号需要连同名称读：`nonlinear_load_v12` 与 `general_ood_v12`
不是同一研究；各目录的 frozen runtime 也不是当前根目录代码的别名。

## 2. 方法在代码中实际做什么

### 2.1 BAPR 惩罚与 belief

入口：[csbapr/agent.py](csbapr/agent.py)。当前根目录的训练流程是：

1. replay 批次训练 actor/critic；若调用方提供近期 rollout，surprise 可用近期数据。
2. surprise 取奖励偏差、Q ensemble 标准差正向突增、当前/目标 critic 参数范数差的最大值，再 EMA 平滑。
3. [BeliefTracker](csbapr/belief/tracker.py) 更新运行长度分布，而非重力/摩擦等环境类别的分布。
4. 运行长度通过 `w = sum_h rho(h) * exp(-0.1*h)` 映射为惩罚强度。
5. warmup 后 actor 系数为 `beta = -2 - penalty_scale*w`，优化
   `Q_mean + beta*Q_std - alpha*log_prob`；具体 frozen 实验有固定 beta 或独立 readout。

当前运行长度滤波先施加 hazard 转移，再用固定 half-normal 似然更新；
长运行段方差更小，surprise 大时偏向短段，末桶保留超出上限的质量。
这不是完整 BOCD 参数学习，也不是隐环境参数的贝叶斯识别。
运行长度由 `update()` 推进，需与环境物理时间区别。

关键区别：根目录 belief 主要改变训练惩罚；普通 `get_action()` 并不
把运行长度 posterior 作为环境类型输入。在线评估中的动作适应，更多来自
另外的 context 输入。二者不能合称一个已经验证的环境识别后验。

### 2.2 Context

[csbapr/context.py](csbapr/context.py) 用近期转移构造
`[delta_state, abs(delta_state), action, reward, norm(delta_state), norm(action)]`，
长期统计归一化后取短期 EMA，拼入 actor/critic。不是 VAE，也不是 meta-learned latent。
replay 保存采样时的前/后 context，而非用今天的 context 重标历史转移。
根目录现已保存、恢复 context 归一化状态。

### 2.3 符号学习

[csbapr/sindy/](csbapr/sindy/) 先从 `(state, action, increment)` 拟合动力学。
**动力学输出不是动作，动力学 Jacobian 不是策略 Jacobian。**
需要任务特定的控制构造才能得到 action-valued `pi_sym`：
LQR 使用拟合矩阵和任务成本解 Riccati 方程；负载任务使用相应反馈构造。
通用 Hopper 尚没有经过性能验证的通用符号动作参考构造。

[csbapr/losses/jacobian.py](csbapr/losses/jacobian.py) 对齐
`D_state pi_deterministic` 与 `D_state pi_sym`，用全动作维平方 Frobenius 差。
它主要约束策略如何随输入变化，不自动消除动作常量偏置，也不直接约束采样噪声。
另有 action-value anchor，但它与 Jacobian 项应分别分析。

critic 的符号项是 detached 的批次 Jacobian 差；actor 项直接回传梯度。
根目录 context 分支不启用这两类符号损失，故不少后续 context 实验实际上
没有在测试 SINDy/Jacobian。负载正结果的 Jacobian-only 版本保留 SINDy，
关闭 NAU 和 critic 符号项，并非“所有符号学习都删掉”。

IRM-inspired selector 按跨环境预测误差方差选择模型，不是因果发现证书。
NAU、SINDy、IRM 三者同时必要的证据目前没有形成。

## 3. 值得保留的有效版本

### 3.1 非线性负载 v11/v12

candidate：`csbapr-centered-mlp-jacobian-only-no-target-shift`。
centered tanh MLP，无 NAU，critic symbolic weight0，actor Jacobian weight0.01。
ID severity1 训练，非线性执行器、持续 Pareto/background 负载，当前负载可观测。
每组100k learner interactions；candidate 另有10k辨识和1k验证交互。

独立训练 seed7/8，再用100条新扰动路径确认。50x 平均成本，越低越好：

| 方法 | v12 ID | v12 50x |
|---|---:|---:|
| Jacobian-only | 0.234204 | 3.607319 |
| No-symbolic | 0.373706 | 5.600672 |
| BAPR | 17.763856 | 230.667860 |
| RE-SAC | 13.743351 | 418.463670 |

相对自身拟合反馈，50x 额外收益约1.36%；不是全部差距都来自 RL。
v13 扩大初始状态半径后，仍胜这些学习器，但半径1.0时相对直接参考
ID/OOD 成本反而高4.06%/2.12%。

证据：[v11](docs/nonlinear_load_v11_replication_results.md)、
[v12](docs/nonlinear_load_v12_paths_results.md)、
[v13](docs/nonlinear_load_v13_radius_results.md)。

### 3.2 持续负载 v31 与参考改进 v35/v36

v31 只开关 Jacobian weight0/0.01，fresh seed21/22、配对前缀、相同预算。
半径1.0，50x 成本由237.95/294.28降至54.25/54.20，下降77.20%/81.58%。
所有12块成本改善；相对自身直接参考，OOD 只再改善0.40--0.66%，ID全块更差。

v35 改为 cost-linear reference，50x 对旧 cubic-local learner 改善5.40--7.00%。
v36 纠正 evaluator 把保存的 clip reference 重建为 tanh 的问题，确认 actor-arm
收益保留，但 ID 仍未通过 own-reference added-value gate。
硬 clip 参考也不能直接套用全局光滑 Jacobian 证书。

证据：[v31](docs/nonlinear_load_v31_symbolic_results.md)、
[v35](docs/nonlinear_load_v35_reference_intervention_results.md)、
[v36](docs/nonlinear_load_v36_reference_alignment_results.md)。

## 4. 失败路线究竟说明什么

### 4.1 跨环境与原始学习器

- v37：三训练 seed 的 Hopper/Pendulum 迁移比较没有通过准入条件。
  context checkpoint 当时漏保存归一化状态，不能把其负结果等同于正确恢复后的性能。
- v38：修正运行长度方向在 Pendulum 有平均收益，但最差 profile 回退；Hopper 有 ID 崩溃，
  某些回合甚至未经历第一次 OOD 切换。数学方向修正不等于训练稳定性修复。
- 修复后的 LQR v2：50x 成本相对 BAPR/RE-SAC 降27.0%/29.9%，但5x更差；
  20x/50x 还差于零动作诊断，不能只用弱学习器相对胜利来证明好控制。

证据：[v37](docs/general_ood_v37_cross_env_results.md)、
[v38](docs/general_ood_v38_belief_results.md)、[LQR v2](docs/lqr_feedback_v2_results.md)。

### 4.2 改善检测器为什么没得到控制收益

- v52：正常物理下，仅继续学习就让冻结校准的 risk 接近饱和；并非环境变了。
- v58：adaptive beta 有局部 return 改善，但静态更强 penalty 也能改善，优势不独立。
- v70：真物理模式在观察上可分，但模型误差大于模式间 gap，尾部仍误判正常。
- v95：冻结 actor 的检测优势不能可靠转移到正在学习的 actor。
- v98--v118：支持覆盖、条件均值、协方差、分类器等改善了部分识别；v118早期准确率98.83%。
- v119：更好识别接回 actor penalty 后，没有相对固定 beta 的整体收益；正常环境有不利 seed。
- v120：当时 source 速度仅约0.04m/s，回报约1040主要来自健康存活奖励；不是好步行策略。

结论是检测、risk calibration、惩罚控制和运动能力必须分开验证。
已知模式的 empirical class weight 也不是未知 OOD 的物理后验。

证据：[v52](docs/general_ood_v52_matched_continuation_results.md)、
[v70](docs/general_ood_v70_physical_audit_results.md)、
[v95](docs/general_ood_v95_fresh_seed_results.md)、
[v118](docs/general_ood_v118_frozen_classifier_confirmation_results.md)、
[v119](docs/general_ood_v119_actor_feedback_results.md)、
[v120](docs/general_ood_v120_mechanism_results.md)。

### 4.3 Source 学习、entropy、critic 与方差

v122 去掉 target L1 uplift 后恢复运动和合理 Q scale，但只一部分 seed 稳定。
v128/v129 mean retention 对已有强 source 有明显收益；v130扩展到新 seed3/4后
没有复现同样生存改善，一支 Q/alpha发散，另一支没有学会好步态。

v132--v134 的 MSE source 最后在正常 Hopper 可靠步行：seed3/4平均回报
3507.69/3286.03，新增 sampled 各60回合全部完整存活。
v135/v136 OOD context 消融却仍全部摔倒；context 支持正常步态，不等于 OOD 适应。
v137--v143 固定 profile、分解物理因素、调 actor/critic 学习率及延长预算
仍未解除极端重力/摩擦下的生存瓶颈，故转向原生 SAC 校准。

证据：[v130](docs/general_ood_v130_independent_replication_results.md)、
[v134](docs/general_ood_v134_frozen_source_robustness_results.md)、
[v136](docs/general_ood_v136_warmup_context_results.md)、
[v143](docs/general_ood_v143_terminal_trace_results.md)。

## 5. 最新 v144--v150：标准 SAC 校准，不是 CS-BAPR 成果

使用 SB3 SAC2.7.0，原生 Hopper-v4 reward、motor bounds、健康终止和1000-step cap。
这一路没有 belief、context、SINDy/Jacobian 或 reward normalization。
“对照”是同 seed 的 no-pulse SAC，而非 RE-SAC 或 BAPR。
2.6591479484724942x 是实际重力倍率；此前 profile intensity50不是每个物理量50倍。

v144--v148：ID 可学、2.0x短冲击常可存活，2.659x的表现与 seed/冲击时点强相关；
从零随机冲击训练可损伤 ID。不能据此称极端重力物理上不可能控制。

v149 从两个稳定的 no-pulse600k完整训练状态，各分叉正常续训/随机恢复续训100k。
actor、critic、Adam、entropy、normal replay、MuJoCo状态与RNG全部恢复。
训练50%回合分配2.0x重力10-step脉冲，onset80--320，仍保留650k/700k两快照。
正常700k全存活，极端 OOD未形成稳定随机训练收益。

| 700k，sampled，2.659x持续10步 | after100完整存活 | after300完整存活 |
|---|---:|---:|
| seed5 control | 4/15 | 3/15 |
| seed5 randomized | 4/15 | 15/15 |
| seed6 control | 10/15 | 15/15 |
| seed6 randomized | 0/15 | 15/15 |

100-step恢复不等于1000-step完整生存；seed5 randomized早期存活从650k的14/15
降到700k的4/15；50-step和永久冲击均无完整存活。训练peak transitions仅540/550条，
约占最终700k replay的0.077/0.079%，稀释是待验证解释，不是已证原因。

### v150：最新完成的冻结噪声时段诊断

固定四个最终700k策略，fresh reset/noise seeds，1200回合，零训练/更新。
每种噪声程序有同场运行的无物理冲击对照；这些对照全部完整存活。
after100冲击的结果如下；prefix quiet 从第一次冲击动作起就恢复正常随机采样。

| 策略 | native | 全程静默 | 仅冲击前静默 | 仅冲击期静默 | 仅恢复期静默 |
|---|---:|---:|---:|---:|---:|
| seed5 control | 0/15 | 5/5 | 4/15 | 4/15 | 2/15 |
| seed5 randomized | 0/15 | 1/5 | 3/15 | 3/15 | 0/15 |
| seed6 control | 11/15 | 5/5 | 15/15 | 11/15 | 12/15 |
| seed6 randomized | 3/15 | 5/5 | 15/15 | 0/15 | 0/15 |

seed6的主要效果来自前冲击轨迹，而非冲击后关噪声。仍需区分步态相位、姿态分布、
mean-policy和噪声路径依赖。seed5未全面修复；其control在after300的全程静默为0/5，
native为1/15，故全局deterministic也不是通用解法。
本诊断使用真实冲击时钟，不是可部署 detector/control，不支持 CS-BAPR 优越性。
与v149不同的是评估随机种子，不是策略训练改变。

证据：[v149](docs/general_ood_v149_warm_recovery_results.md)、
[v150协议](docs/general_ood_v150_noise_timing_protocol.md)、
[v150结果](docs/general_ood_v150_noise_timing_results.md)、
[v150小型原始结果与汇总](results/general_ood_v150_noise_timing/summary.json)。
最后四任务：t152176--t152179，均exit0；没有再派新一轮训练。

## 6. 理论与论文：正确部分、未接通部分

当前主稿：[paper/paper.tex](paper/paper.tex)，附录：[paper/appendix.tex](paper/appendix.tex)。

正确的条件性输出界是 `delta + epsilon*||d|| + 0.5*L_err*||d||^2`。
合成规则已修为 `L_H*K_G^2 + K_H*L_G`，包括输出tanh的曲率。
孤立NAU线性、光滑actor可有确定性全策略导数界；这些计算本身不能保证RL回报。
frozen actor/belief/penalty的Bellman contraction也不是共同变化的深度actor-critic收敛证明。

未接通的链条：有限训练损失如何给出OOD段的点态导数误差；动力学辨识怎样产生
适用于通用任务的动作参考；确定性动作误差怎样转为带噪、长时域、硬终止的控制收益。
需求倍率不是状态位移，输出误差单位不是return；不能把理论曲线覆盖在经验return上。

当前公交主稿已收缩为稳定化recipe和条件性输出分析；SINDy/IRM不在主结果中启用。
历史LeakyReLU actor未获全策略证书；修复后的smooth-nau也未被历史表格评估。
本研究现在应选择“可信符号参考下的结构正则”还是“非平稳潜在环境适应”，
二者不能仅凭同一个名称拼成一条已有实验闭环的贡献。

## 7. 已确认的问题与验证边界

1. **原始公交逐seed证据缺失。** 稿件CSV是聚合数值转录，不能重建置信区间、
   配对统计、checkpoint选择和线性拟合诊断；后续清楚记录的实验不能替代旧证据。
2. **优势主要来自哪里仍需界定。** 公交主要是稳定化包；负载任务主要接近拟合反馈；
   更多模式识别尚未变成稳定控制收益。需要与直接参考、no-symbolic和强标准SAC分别对照。
3. **额外预算不相等。** candidate有辨识/验证交互；既应报learner预算，也应报总环境预算。
   warm fork共享源训练，不能按四个独立700k训练seed计算；评估重复不是训练seed。
4. **已知模式和弱source局限。** 一部分likelihood数据使用已知模式/仿真模板；
   多轮readout结论只条件于一个source。高准确率不能外推未知极端环境或独立训练复现。
5. **在线学习混入环境识别。** reference、行为覆盖、reward尺度和critic参数都在变；
   正常环境也会触发risk。belief运行长度、模式posterior和departure risk应分开定义。
6. **恢复、运动、回报不是同一指标。** standing可拿健康奖励，短恢复后仍可摔倒；
   仅相对更差的baseline取胜不足以证明可用控制。
7. **可控性与可学习性不能混淆。** 常重力失败、动作饱和统计或关节姿态不足以证明不可控；
   极端永久变化、瞬时脉冲和重尾负载也不是同一种OOD。
8. **证书对象不一致。** actor类型、reference的tanh/clip、采样std和理论正则对象必须明确；
   原始架构无证书，光滑新架构有保守输出界，均不等于return或安全性保证。
9. **历史说明有过时内容。** `paper/GPT_review.md`和`proof_preview.md`不是当前证据，
   其中Lean机器验证/ReLU无法多项式外推等说法不能当成已成立结论。
   `csbapr/irm/causal_filter.py`仍有对应不存在Lean文件、因果保证的旧注释。
   当前仓库没有`.lean`、`lakefile`或`lean-toolchain`工程。
10. **工程复现边界。** root与各frozen runtime不同；服务器绝对路径和依赖环境
    不随Git自动可用。checkpoint/replay在服务器，不能以Git源码归档宣称每个实验可立即重跑。

本轮确认正确的部分：当前主稿临时目录三遍pdflatex构建成功，23页；
无undefined citation/reference或overfull box，仅一处非致命underfull行。
43项针对belief、env seed、符号接口、context恢复、v150动作与结果验证的测试通过；
根目录Python语法编译和修改过的v37启动脚本语法检查通过。
这不是全测试集通过，也不是证明全部历史实验、理论或性能有效。
本轮没有重新联网逐条核对参考文献；既有记录见
[reference_audit.csv](paper/reference_audit.csv)和[manuscript_audit.md](paper/manuscript_audit.md)。

## 8. 请 GPT6 优先作出的判断

1. **研究身份**：是否应将负载型symbolic-reference regularization作为独立受限贡献，
   与通用潜在环境适应分开？哪些公交结论能够保留，哪些必须重跑后才能作为主结果？
2. **环境合同**：怎样保留极端重尾随机性，又有可观测、可操作、有限控制能力的任务？
   正常可学、冲击可恢复、恢复后仍可持续控制，应如何各自验证？
3. **belief的作用**：现有scalar pessimism是否足够？是否需要明确动力学参数/模式的后验，
   并让actor在推理时条件化？怎样排除学习漂移、reference偏差和policy导致的假环境变化？
4. **符号机制的必要性**：当直接拟合反馈已经很强，Jacobian约束是否只是控制器蒸馏？
   最小哪一项对比能区分结构外推、参考先验、额外数据和RL的贡献？
5. **通用参考构造**：不能把dynamics直接作policy，除了任务特定LQR/反馈，
   是否存在值得验证的局部控制构造？它需要哪些可检验条件，成本是否抵消收益？
6. **v150后续**：先做同物理状态/不同动作噪声的branch实验，还是gait-phase conditioning？
   怎样设计一个无真实时钟、仅观测历史的因果干预，而非从oracle窗口偷用答案？
7. **baseline与预算**：建议一组最小强对照，统一总交互、架构/优化器、源状态、
   独立训练seed和最终checkpoint。如何把弱baseline胜利与有效控制分开？
8. **可写的理论**：保留哪一个条件性定理，补哪一个最小桥接结果，
   才能支撑最终方法，而不是继续叠加必要性、因果性和收敛性大主张？

希望返回：明确选一条主路线，列出已证/待证/应放弃的主张；给出一轮最小实验
及成功、失败后的不同动作，说明为什么它能改变研究决策。不是一份继续扫参数的清单。

## 9. 后续执行约束

- 保留负结果和所有预声明快照；不用held-out结果挑checkpoint、阈值或有利seed。
- 不把训练seed、共享source的continuation seed、环境seed和reset/noise重复混为一类。
- 服务器只小拉取：log优先于CSV，CSV/小JSON优先于checkpoint；大型状态留服务器。
- 调度选择可用Linux节点，不派Windows，不加机器/GPU偏好。当前jtl110gpu根盘仍满。
- 保持原生reward、action范围、健康终止等合同；改变环境应先验证环境合同，再观察算法排序。
- root的探索改动与冻结历史runtime分别保存；不会因本次push重写既有实验。
- 默认继续研究时先读取本交接和相应protocol/results/summary；当前无新轮训练待派。

完整证据入口：`docs/*_protocol.md`、`docs/*_results.md`、
`experiments/*/dispatch_manifest.json`及`results/*/summary.json`。
报告中的历史running/next-step描述是当时快照，最新v150状态已更新为complete。
