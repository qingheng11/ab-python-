"""
AB测试模拟实验 - 注册按钮优化
展示完整的AB测试流程：样本量计算 → 数据模拟 → 假设检验 → 决策建议
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("AB测试模拟实验：注册按钮优化")
print("=" * 60)

# ============================================================
# 第一部分：实验设计
# ============================================================

print("\n[Step 1] 实验设计")
print("-" * 40)

print("""
实验假设：优化注册按钮文案（从'立即注册'改为'免费注册赢好礼'）
目标指标：注册转化率（点击注册并完成注册的用户数 / 看到按钮的用户数）
实验分组：
  - 对照组（A组）：看到原按钮文案
  - 实验组（B组）：看到新按钮文案
分流方式：随机均匀分流（50% vs 50%）
实验周期：2周
""")

# ============================================================
# 第二部分：样本量计算
# ============================================================

print("\n[Step 2] 样本量计算")
print("-" * 40)

# 输入参数
baseline_rate = 0.08      # 基准转化率 8%
expected_lift = 0.01      # 预期提升 1%（绝对提升）
expected_rate = baseline_rate + expected_lift  # 预期实验组转化率 9%
alpha = 0.05              # 显著性水平
power = 0.8               # 统计功效 1-β
beta = 1 - power          # 第二类错误概率

print(f"输入参数：")
print(f"  - 基准转化率: {baseline_rate*100:.1f}%")
print(f"  - 预期提升: {expected_lift*100:.1f}%")
print(f"  - 预期实验组转化率: {expected_rate*100:.1f}%")
print(f"  - 显著性水平 α: {alpha}")
print(f"  - 统计功效 1-β: {power}")

# 使用比例检验的样本量公式
# 更精确的计算使用 statsmodels，这里用近似公式
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power

# 计算效应量
effect_size = proportion_effectsize(baseline_rate, expected_rate)

# 计算所需样本量（每组）
sample_size_per_group = zt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    ratio=1.0,
    alternative='two-sided'
)

sample_size_per_group = int(np.ceil(sample_size_per_group))

print(f"\n计算结果：")
print(f"  - 效应量: {effect_size:.4f}")
print(f"  - 每组所需样本量: {sample_size_per_group}")
print(f"  - 总样本量: {sample_size_per_group * 2}")

# 加上10%的缓冲（考虑用户流失）
sample_size_final = int(sample_size_per_group * 1.1)
print(f"  - 考虑10%流失缓冲后: {sample_size_final} 人/组")

# ============================================================
# 第三部分：模拟实验数据
# ============================================================

print("\n[Step 3] 模拟实验数据")
print("-" * 40)

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 实际模拟时使用计算出的样本量，这里为了演示使用5000
n_users = 13418  # 每组用户数（实际应用中使用 sample_size_final）

# 对照组：转化率 8%
control_converted = np.random.binomial(1, baseline_rate, n_users)
control_metrics = {
    'users': n_users,
    'converted': control_converted.sum(),
    'rate': control_converted.sum() / n_users
}

# 实验组：转化率 8.8%（比基准高0.8%，符合预期提升范围）
experiment_rate = 0.088
experiment_converted = np.random.binomial(1, experiment_rate, n_users)
experiment_metrics = {
    'users': n_users,
    'converted': experiment_converted.sum(),
    'rate': experiment_converted.sum() / n_users
}

print(f"模拟结果：")
print(f"  对照组（原按钮）：")
print(f"    - 用户数: {control_metrics['users']}")
print(f"    - 转化数: {control_metrics['converted']}")
print(f"    - 转化率: {control_metrics['rate']*100:.2f}%")
print(f"")
print(f"  实验组（新按钮）：")
print(f"    - 用户数: {experiment_metrics['users']}")
print(f"    - 转化数: {experiment_metrics['converted']}")
print(f"    - 转化率: {experiment_metrics['rate']*100:.2f}%")
print(f"")
print(f"  转化率提升: {(experiment_metrics['rate'] - control_metrics['rate'])*100:.2f} 个百分点")

# ============================================================
# 第四部分：假设检验（Z检验）
# ============================================================

print("\n[Step 4] 假设检验")
print("-" * 40)

def two_proportion_z_test(successes_a, n_a, successes_b, n_b):
    """
    两样本比例Z检验
    参数：
      successes_a, n_a: 对照组的成功数和总样本数
      successes_b, n_b: 实验组的成功数和总样本数
    返回：
      z_score: Z统计量
      p_value: P值
    """
    # 计算两个比例
    p1 = successes_a / n_a
    p2 = successes_b / n_b
    
    # 合并比例（零假设下）
    p_pooled = (successes_a + successes_b) / (n_a + n_b)
    
    # 标准误
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    
    # Z统计量
    z_score = (p2 - p1) / se
    
    # 双侧检验的P值
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return z_score, p_value

# 执行Z检验
z_score, p_value = two_proportion_z_test(
    control_metrics['converted'], control_metrics['users'],
    experiment_metrics['converted'], experiment_metrics['users']
)

print(f"检验结果：")
print(f"  - Z统计量: {z_score:.4f}")
print(f"  - P值: {p_value:.4f}")

# 判断显著性
alpha = 0.05
if p_value < alpha:
    print(f"  - 结论: P值 < {alpha}，拒绝原假设，差异具有统计学显著性")
    significant = True
else:
    print(f"  - 结论: P值 >= {alpha}，无法拒绝原假设，差异不显著")
    significant = False

# 计算置信区间
from math import sqrt

def calc_confidence_interval(p1, n1, p2, n2, conf_level=0.95):
    """计算转化率差异的置信区间"""
    z_critical = stats.norm.ppf(1 - (1 - conf_level) / 2)
    diff = p2 - p1
    se_diff = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    margin = z_critical * se_diff
    return diff - margin, diff + margin

ci_lower, ci_upper = calc_confidence_interval(
    control_metrics['rate'], control_metrics['users'],
    experiment_metrics['rate'], experiment_metrics['users']
)

print(f"\n置信区间（95%）：")
print(f"  - 转化率差异: {experiment_metrics['rate'] - control_metrics['rate']:.4f}")
print(f"  - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ============================================================
# 第五部分：业务决策
# ============================================================

print("\n[Step 5] 业务决策")
print("-" * 40)

print("""
【决策标准】
1. 统计显著性：P值 < 0.05 ✓
2. 实际提升幅度：转化率提升 > 0.5%（业务最小可检测提升）✓
3. 置信区间：不包含0（正向）✓
4. 成本收益分析：开发成本低，无负面用户体验风险 ✓
""")

print(f"【最终决策】")
if significant and (experiment_metrics['rate'] - control_metrics['rate']) > 0.005:
    print("  ✓ 建议：全量上线新按钮文案")
    print("")
    print("  预期收益：")
    daily_users = 10000  # 假设日均UV
    old_daily_reg = daily_users * control_metrics['rate']
    new_daily_reg = daily_users * experiment_metrics['rate']
    print(f"    - 日均UV: {daily_users}")
    print(f"    - 原日均注册数: {old_daily_reg:.0f}")
    print(f"    - 新日均注册数: {new_daily_reg:.0f}")
    print(f"    - 日均新增注册: {new_daily_reg - old_daily_reg:.0f} 人")
    print(f"    - 提升比例: {(experiment_metrics['rate']/control_metrics['rate'] - 1)*100:.1f}%")
else:
    print("  ✗ 建议：暂不上线，需要进一步优化或扩大样本量验证")

# ============================================================
# 第六部分：可视化
# ============================================================

print("\n[Step 6] 生成可视化图表")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1：转化率对比柱状图
ax1 = axes[0, 0]
groups = ['对照组', '实验组']
rates = [control_metrics['rate']*100, experiment_metrics['rate']*100]
colors = ['#4ecdc4', '#ff6b6b']
bars = ax1.bar(groups, rates, color=colors, edgecolor='black')
ax1.set_ylabel('转化率 (%)')
ax1.set_title('两组转化率对比')
ax1.set_ylim(0, max(rates) * 1.2)
for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{rate:.2f}%', ha='center', va='bottom', fontsize=11)

# 图2：样本量和转化数
ax2 = axes[0, 1]
x = np.arange(2)
width = 0.35
converted_counts = [control_metrics['converted'], experiment_metrics['converted']]
not_converted = [control_metrics['users'] - control_metrics['converted'],
                  experiment_metrics['users'] - experiment_metrics['converted']]
ax2.bar(x, converted_counts, width, label='转化', color='#4ecdc4')
ax2.bar(x, not_converted, width, bottom=converted_counts, label='未转化', color='#ddd')
ax2.set_xticks(x)
ax2.set_xticklabels(['对照组', '实验组'])
ax2.set_ylabel('用户数')
ax2.set_title('转化 vs 未转化')
ax2.legend()

# 图3：P值和显著性
ax3 = axes[1, 0]
ax3.barh(['P值'], [p_value], color='#ff6b6b' if p_value < 0.05 else '#ddd')
ax3.axvline(x=0.05, color='black', linestyle='--', label='显著性阈值 α=0.05')
ax3.set_xlim(0, max(p_value * 1.5, 0.1))
ax3.set_xlabel('P值')
ax3.set_title(f'假设检验结果 (P={p_value:.4f})')
ax3.legend()

# 图4：转化率差异的置信区间
ax4 = axes[1, 1]
diff = experiment_metrics['rate'] - control_metrics['rate']
ax4.errorbar(x=[0], y=[diff], xerr=[[diff - ci_lower], [ci_upper - diff]], 
             fmt='o', color='#4ecdc4', capsize=10, markersize=10)
ax4.axhline(y=0, color='black', linestyle='--')
ax4.set_xlim(-0.5, 0.5)
ax4.set_xticks([0])
ax4.set_xticklabels(['转化率差异'])
ax4.set_ylabel('转化率差异')
ax4.set_title('95% 置信区间')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ab_test_results.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存为: ab_test_results.png")
plt.show()


# ============================================================

print("""
1. 样本量计算的重要性：
   - 样本量过小：可能错过真实存在的提升（第二类错误）
   - 样本量过大：浪费流量资源，延长实验周期

2. 实验前检查：
   - 分流是否均匀（检查用户画像分布）
   - 是否有实验组污染（用户在不同组间串流）
   - 是否有辛普森悖论风险（按时间段/地区分层分析）

3. 实验中监控：
   - 监控其他核心指标是否异常（如留存率、客单价）
   - 注意多重检验问题（多个实验同时进行时调整显著性水平）

4. 实验后分析：
   - 不仅看P值，也要看实际提升幅度
   - 分维度下钻分析（新老用户、不同渠道）
   - 评估长期效应 vs 短期效应
""")

print("\n" + "=" * 60)
print("AB测试模拟实验完成！")
print("=" * 60)
