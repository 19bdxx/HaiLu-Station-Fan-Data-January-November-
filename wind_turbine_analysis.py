#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三台风机全量数据分析脚本
2023年1-11月三台风机数据分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class WindTurbineAnalyzer:
    """风机数据分析器"""
    
    # 分析参数常量
    RATED_POWER = 2500  # 额定功率 (kW)
    MAX_EXPECTED_POWER = 3000  # 最大期望功率 (kW)
    OPERATING_THRESHOLD = 50  # 运行状态判定阈值 (kW)
    HIGH_WIND_THRESHOLD = 5  # 高风速阈值 (m/s)
    LOW_POWER_THRESHOLD = 100  # 低功率阈值 (kW)
    LOW_EFFICIENCY_QUANTILE = 0.1  # 低效运行分位数阈值
    POWER_CHANGE_THRESHOLD = 1000  # 功率突变阈值 (kW)
    WIND_SPEED_EPSILON = 1  # 风速计算中的小量，避免除零
    
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.turbines = {}
        self.report_content = []
        self.output_dir = 'analysis_results'
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_data(self):
        """加载三台风机的数据"""
        print("正在加载数据...")
        
        for i in range(1, 4):
            filename = f'{i}__风机数据.csv'
            filepath = os.path.join(self.data_dir, filename)
            
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                # 转换时间列
                df['统计时间'] = pd.to_datetime(df['统计时间'], format='%Y/%m/%d %H:%M')
                
                self.turbines[i] = df
                print(f"风机{i}数据加载成功: {len(df)} 条记录")
            except Exception as e:
                print(f"加载风机{i}数据时出错: {e}")
        
        return self.turbines
    
    def clean_data(self):
        """数据清洗和预处理"""
        print("\n正在进行数据清洗...")
        
        cleaning_stats = {}
        
        for turbine_id, df in self.turbines.items():
            original_count = len(df)
            
            # 统计缺失值
            missing_count = df.isnull().sum().sum()
            
            # 计算功率列（核心指标）
            power_col = f'#{turbine_id}_平均网侧有功功率(kW)'
            
            if power_col in df.columns:
                # 统计异常值（负功率或异常大的功率）
                valid_power = df[power_col].notna()
                negative_power = (df[power_col] < 0).sum()
                abnormal_high = (df[power_col] > self.MAX_EXPECTED_POWER).sum()  # 使用常量
                
                cleaning_stats[turbine_id] = {
                    '总记录数': original_count,
                    '缺失值数': missing_count,
                    '负功率记录': negative_power,
                    '异常高功率记录': abnormal_high,
                    '有效数据率': f"{(valid_power.sum() / len(df) * 100):.2f}%"
                }
            
            # 填充缺失值（使用前向填充和后向填充）
            # 注意：这种方法假设数据是连续的，适用于短暂的数据采集中断
            # 对于长时间停机或传感器故障，应在异常检测中单独处理
            df_cleaned = df.ffill().bfill()
            self.turbines[turbine_id] = df_cleaned
        
        print("数据清洗完成")
        return cleaning_stats
    
    def calculate_statistics(self):
        """计算各风机的统计指标"""
        print("\n正在计算统计指标...")
        
        stats = {}
        
        for turbine_id, df in self.turbines.items():
            power_col = f'#{turbine_id}_平均网侧有功功率(kW)'
            wind_speed_col = f'#{turbine_id}_平均风速(m/s)'
            
            if power_col in df.columns:
                # 计算发电量 (kWh) - 每10分钟采样
                total_energy = df[power_col].sum() * (10/60)  # 转换为小时
                
                # 运行时长（功率>阈值的时间）
                operating_hours = (df[power_col] > self.OPERATING_THRESHOLD).sum() * (10/60)
                
                # 平均功率
                avg_power = df[power_col].mean()
                max_power = df[power_col].max()
                
                # 利用小时数（基于额定功率）
                utilization_hours = total_energy / self.RATED_POWER
                
                # 容量系数（实际发电量与理论最大发电量的比值）
                total_hours = len(df) * (10/60)
                capacity_factor = (total_energy / (self.RATED_POWER * total_hours) * 100) if total_hours > 0 else 0
                
                # 风速统计
                avg_wind_speed = df[wind_speed_col].mean() if wind_speed_col in df.columns else 0
                
                # 停机时长（功率低于阈值的时间）
                downtime_hours = (df[power_col] < self.OPERATING_THRESHOLD).sum() * (10/60)
                
                stats[turbine_id] = {
                    '总发电量(MWh)': total_energy / 1000,
                    '平均功率(kW)': avg_power,
                    '最大功率(kW)': max_power,
                    '运行时长(小时)': operating_hours,
                    '停机时长(小时)': downtime_hours,
                    '利用小时数': utilization_hours,
                    '容量系数(%)': capacity_factor,
                    '平均风速(m/s)': avg_wind_speed,
                    '可用率(%)': (operating_hours / total_hours * 100)
                }
        
        return stats
    
    def detect_anomalies(self):
        """检测异常和潜在问题"""
        print("\n正在检测异常...")
        
        anomalies = {}
        
        for turbine_id, df in self.turbines.items():
            power_col = f'#{turbine_id}_平均网侧有功功率(kW)'
            wind_speed_col = f'#{turbine_id}_平均风速(m/s)'
            
            turbine_anomalies = {
                '异常停机': [],
                '低效运行': [],
                '数据异常': []
            }
            
            if power_col in df.columns and wind_speed_col in df.columns:
                # 检测异常停机（风速高但功率低）
                high_wind_low_power = (df[wind_speed_col] > self.HIGH_WIND_THRESHOLD) & (df[power_col] < self.LOW_POWER_THRESHOLD)
                if high_wind_low_power.sum() > 0:
                    anomaly_times = df[high_wind_low_power]['统计时间'].tolist()
                    turbine_anomalies['异常停机'] = anomaly_times[:10]  # 仅记录前10个
                
                # 检测低效运行（功率系数异常低）
                # 效率指标 = 功率 / (风速² + ε)，其中ε避免除零
                # 简化判断：在相同风速下，功率明显偏低
                df['效率指标'] = df[power_col] / (df[wind_speed_col] ** 2 + self.WIND_SPEED_EPSILON)
                efficiency_threshold = df['效率指标'].quantile(self.LOW_EFFICIENCY_QUANTILE)
                low_efficiency = df['效率指标'] < efficiency_threshold
                
                if low_efficiency.sum() > 0:
                    low_eff_times = df[low_efficiency]['统计时间'].tolist()
                    turbine_anomalies['低效运行'] = low_eff_times[:10]
                
                # 检测数据异常（功率突变）
                df['功率变化'] = df[power_col].diff().abs()
                sudden_changes = df['功率变化'] > self.POWER_CHANGE_THRESHOLD  # 使用常量
                
                if sudden_changes.sum() > 0:
                    change_times = df[sudden_changes]['统计时间'].tolist()
                    turbine_anomalies['数据异常'] = change_times[:10]
            
            anomalies[turbine_id] = turbine_anomalies
        
        return anomalies
    
    def visualize_data(self):
        """生成数据可视化图表"""
        print("\n正在生成可视化图表...")
        
        # 1. 发电量趋势对比图
        self._plot_power_trends()
        
        # 2. 风速与功率散点图
        self._plot_wind_power_relationship()
        
        # 3. 日发电量对比
        self._plot_daily_energy()
        
        # 4. 月度统计对比
        self._plot_monthly_statistics()
        
        # 5. 可用率和容量系数对比
        self._plot_performance_metrics()
        
        print("可视化图表生成完成")
    
    def _plot_power_trends(self):
        """绘制功率趋势图"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        for idx, turbine_id in enumerate([1, 2, 3]):
            df = self.turbines[turbine_id]
            power_col = f'#{turbine_id}_平均网侧有功功率(kW)'
            
            if power_col in df.columns:
                # 按天重采样
                daily_data = df.set_index('统计时间')[power_col].resample('D').mean()
                
                axes[idx].plot(daily_data.index, daily_data.values, linewidth=1, label=f'风机{turbine_id}')
                axes[idx].set_title(f'风机{turbine_id}日平均功率趋势', fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('平均功率 (kW)', fontsize=10)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend()
        
        axes[2].set_xlabel('日期', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_功率趋势图.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_wind_power_relationship(self):
        """绘制风速与功率关系图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, turbine_id in enumerate([1, 2, 3]):
            df = self.turbines[turbine_id]
            power_col = f'#{turbine_id}_平均网侧有功功率(kW)'
            wind_speed_col = f'#{turbine_id}_平均风速(m/s)'
            
            if power_col in df.columns and wind_speed_col in df.columns:
                # 采样数据以提高绘图效率
                sample_df = df.sample(min(5000, len(df)))
                
                axes[idx].scatter(sample_df[wind_speed_col], sample_df[power_col], 
                                 alpha=0.3, s=1, c='blue')
                axes[idx].set_title(f'风机{turbine_id}风速-功率关系', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('风速 (m/s)', fontsize=10)
                axes[idx].set_ylabel('功率 (kW)', fontsize=10)
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_风速功率关系图.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_daily_energy(self):
        """绘制日发电量对比图"""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        for turbine_id in [1, 2, 3]:
            df = self.turbines[turbine_id]
            power_col = f'#{turbine_id}_平均网侧有功功率(kW)'
            
            if power_col in df.columns:
                # 计算日发电量
                daily_energy = df.set_index('统计时间')[power_col].resample('D').sum() * (10/60) / 1000
                
                ax.plot(daily_energy.index, daily_energy.values, label=f'风机{turbine_id}', linewidth=1.5)
        
        ax.set_title('三台风机日发电量对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=10)
        ax.set_ylabel('日发电量 (MWh)', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_日发电量对比图.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_statistics(self):
        """绘制月度统计对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        monthly_data = {1: {}, 2: {}, 3: {}}
        
        for turbine_id in [1, 2, 3]:
            df = self.turbines[turbine_id]
            power_col = f'#{turbine_id}_平均网侧有功功率(kW)'
            wind_speed_col = f'#{turbine_id}_平均风速(m/s)'
            
            if power_col in df.columns:
                df_monthly = df.set_index('统计时间')
                
                # 月发电量
                monthly_energy = df_monthly[power_col].resample('ME').sum() * (10/60) / 1000
                monthly_data[turbine_id]['energy'] = monthly_energy
                
                # 月平均功率
                monthly_power = df_monthly[power_col].resample('ME').mean()
                monthly_data[turbine_id]['power'] = monthly_power
                
                # 月平均风速
                if wind_speed_col in df.columns:
                    monthly_wind = df_monthly[wind_speed_col].resample('ME').mean()
                    monthly_data[turbine_id]['wind'] = monthly_wind
        
        # 绘制月发电量
        for turbine_id in [1, 2, 3]:
            if 'energy' in monthly_data[turbine_id]:
                axes[0, 0].plot(monthly_data[turbine_id]['energy'].index, 
                               monthly_data[turbine_id]['energy'].values, 
                               marker='o', label=f'风机{turbine_id}')
        axes[0, 0].set_title('月发电量对比', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('发电量 (MWh)', fontsize=10)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 绘制月平均功率
        for turbine_id in [1, 2, 3]:
            if 'power' in monthly_data[turbine_id]:
                axes[0, 1].plot(monthly_data[turbine_id]['power'].index, 
                               monthly_data[turbine_id]['power'].values, 
                               marker='o', label=f'风机{turbine_id}')
        axes[0, 1].set_title('月平均功率对比', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('平均功率 (kW)', fontsize=10)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 绘制月平均风速
        for turbine_id in [1, 2, 3]:
            if 'wind' in monthly_data[turbine_id]:
                axes[1, 0].plot(monthly_data[turbine_id]['wind'].index, 
                               monthly_data[turbine_id]['wind'].values, 
                               marker='o', label=f'风机{turbine_id}')
        axes[1, 0].set_title('月平均风速对比', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('平均风速 (m/s)', fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 发电量柱状对比
        turbine_ids = [1, 2, 3]
        total_energies = [monthly_data[tid]['energy'].sum() if 'energy' in monthly_data[tid] else 0 
                         for tid in turbine_ids]
        axes[1, 1].bar(['风机1', '风机2', '风机3'], total_energies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_title('总发电量对比', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('总发电量 (MWh)', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(total_energies):
            axes[1, 1].text(i, v + max(total_energies)*0.02, f'{v:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_月度统计对比图.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics(self):
        """绘制性能指标对比图"""
        stats = self.calculate_statistics()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        turbine_ids = [1, 2, 3]
        turbine_names = ['风机1', '风机2', '风机3']
        
        # 可用率对比
        availability = [stats[tid]['可用率(%)'] for tid in turbine_ids if tid in stats]
        axes[0].bar(turbine_names[:len(availability)], availability, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_title('可用率对比', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('可用率 (%)', fontsize=10)
        axes[0].set_ylim([0, 100])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(availability):
            axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
        
        # 容量系数对比
        capacity_factors = [stats[tid]['容量系数(%)'] for tid in turbine_ids if tid in stats]
        axes[1].bar(turbine_names[:len(capacity_factors)], capacity_factors, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_title('容量系数对比', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('容量系数 (%)', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(capacity_factors):
            axes[1].text(i, v + max(capacity_factors)*0.02, f'{v:.1f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_性能指标对比图.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, cleaning_stats, statistics, anomalies):
        """生成分析报告"""
        print("\n正在生成分析报告...")
        
        report = []
        report.append("# 三台风机全量数据分析报告\n")
        report.append(f"**报告生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        report.append("**数据时间范围**: 2023年1月 - 2023年11月\n")
        report.append("\n---\n")
        
        # 1. 数据概况
        report.append("## 1. 数据概况\n")
        report.append("### 1.1 数据加载情况\n")
        for turbine_id in [1, 2, 3]:
            if turbine_id in self.turbines:
                df = self.turbines[turbine_id]
                report.append(f"- **风机{turbine_id}**: {len(df):,} 条记录\n")
        
        report.append("\n### 1.2 数据清洗统计\n")
        if cleaning_stats:
            report.append("| 风机 | 总记录数 | 缺失值数 | 负功率记录 | 异常高功率 | 有效数据率 |\n")
            report.append("|------|---------|---------|-----------|-----------|----------|\n")
            for turbine_id, stats in cleaning_stats.items():
                report.append(f"| 风机{turbine_id} | {stats['总记录数']:,} | {stats['缺失值数']:,} | "
                            f"{stats['负功率记录']} | {stats['异常高功率记录']} | {stats['有效数据率']} |\n")
        
        # 2. 运行统计
        report.append("\n## 2. 风机运行统计\n")
        if statistics:
            report.append("### 2.1 主要性能指标\n")
            report.append("| 指标 | 风机1 | 风机2 | 风机3 |\n")
            report.append("|------|-------|-------|-------|\n")
            
            metrics = ['总发电量(MWh)', '平均功率(kW)', '最大功率(kW)', '运行时长(小时)', 
                      '停机时长(小时)', '利用小时数', '容量系数(%)', '平均风速(m/s)', '可用率(%)']
            
            for metric in metrics:
                row = f"| {metric} |"
                for turbine_id in [1, 2, 3]:
                    if turbine_id in statistics and metric in statistics[turbine_id]:
                        value = statistics[turbine_id][metric]
                        if isinstance(value, float):
                            row += f" {value:.2f} |"
                        else:
                            row += f" {value} |"
                    else:
                        row += " - |"
                report.append(row + "\n")
        
        # 3. 性能对比分析
        report.append("\n## 3. 性能对比分析\n")
        if statistics:
            # 找出最佳和最差性能
            energies = {tid: statistics[tid]['总发电量(MWh)'] for tid in [1, 2, 3] if tid in statistics}
            if energies:
                best_turbine = max(energies, key=energies.get)
                worst_turbine = min(energies, key=energies.get)
                
                report.append(f"### 3.1 发电量分析\n")
                report.append(f"- **最佳表现**: 风机{best_turbine}，总发电量 {energies[best_turbine]:.2f} MWh\n")
                report.append(f"- **最差表现**: 风机{worst_turbine}，总发电量 {energies[worst_turbine]:.2f} MWh\n")
                report.append(f"- **差异**: {((energies[best_turbine] - energies[worst_turbine]) / energies[worst_turbine] * 100):.2f}%\n")
            
            # 可用率对比
            availabilities = {tid: statistics[tid]['可用率(%)'] for tid in [1, 2, 3] if tid in statistics}
            if availabilities:
                report.append(f"\n### 3.2 可用率分析\n")
                for tid, avail in availabilities.items():
                    status = "优秀" if avail > 95 else "良好" if avail > 90 else "需改进"
                    report.append(f"- 风机{tid}: {avail:.2f}% ({status})\n")
        
        # 4. 异常检测结果
        report.append("\n## 4. 异常检测与问题分析\n")
        if anomalies:
            for turbine_id, turbine_anomalies in anomalies.items():
                report.append(f"\n### 4.{turbine_id} 风机{turbine_id}异常情况\n")
                
                for anomaly_type, anomaly_list in turbine_anomalies.items():
                    if anomaly_list:
                        report.append(f"- **{anomaly_type}**: 检测到 {len(anomaly_list)} 个时间点\n")
                        if len(anomaly_list) > 0:
                            report.append(f"  - 示例时间: {anomaly_list[0]}\n")
                    else:
                        report.append(f"- **{anomaly_type}**: 未检测到异常\n")
        
        # 5. 可视化图表
        report.append("\n## 5. 数据可视化\n")
        report.append("本次分析生成了以下可视化图表：\n")
        report.append("1. **功率趋势图** - 展示三台风机的日平均功率变化趋势\n")
        report.append("2. **风速功率关系图** - 展示各风机风速与输出功率的关系\n")
        report.append("3. **日发电量对比图** - 对比三台风机的日发电量\n")
        report.append("4. **月度统计对比图** - 包含月发电量、月平均功率、月平均风速等\n")
        report.append("5. **性能指标对比图** - 对比可用率和容量系数\n")
        report.append("\n所有图表已保存至 `analysis_results` 目录。\n")
        
        # 6. 结论与建议
        report.append("\n## 6. 结论与优化建议\n")
        report.append("### 6.1 主要发现\n")
        
        if statistics:
            # 自动生成结论
            energies = [statistics[tid]['总发电量(MWh)'] for tid in [1, 2, 3] if tid in statistics]
            if len(energies) == 3:
                avg_energy = sum(energies) / len(energies)
                std_energy = np.std(energies)
                
                if std_energy / avg_energy > 0.1:
                    report.append("1. **发电量差异显著**: 三台风机发电量存在较大差异，需要进一步分析原因\n")
                else:
                    report.append("1. **发电量较为均衡**: 三台风机发电量表现相对一致\n")
            
            availabilities = [statistics[tid]['可用率(%)'] for tid in [1, 2, 3] if tid in statistics]
            if availabilities and min(availabilities) < 95:
                report.append("2. **可用率有待提高**: 部分风机可用率低于95%，建议加强维护\n")
            
            capacity_factors = [statistics[tid]['容量系数(%)'] for tid in [1, 2, 3] if tid in statistics]
            if capacity_factors:
                avg_cf = sum(capacity_factors) / len(capacity_factors)
                report.append(f"3. **平均容量系数**: {avg_cf:.2f}%，")
                if avg_cf < 25:
                    report.append("处于较低水平，建议优化运维策略\n")
                elif avg_cf < 35:
                    report.append("处于中等水平\n")
                else:
                    report.append("表现良好\n")
        
        report.append("\n### 6.2 优化建议\n")
        report.append("1. **加强预防性维护**: 根据异常检测结果，制定针对性的维护计划\n")
        report.append("2. **优化运行参数**: 分析高效运行时段的参数设置，推广至其他时段\n")
        report.append("3. **改进故障响应**: 对于检测到的异常停机，建立快速响应机制\n")
        report.append("4. **性能对标**: 将表现最好的风机作为标杆，分析其优势并推广\n")
        report.append("5. **数据质量提升**: 减少数据缺失和异常，提高数据采集可靠性\n")
        report.append("6. **季节性优化**: 根据不同月份的风况特点，调整运维策略\n")
        
        report.append("\n---\n")
        report.append("**注**: 本报告基于2023年1-11月数据生成，实际优化方案需结合现场情况制定。\n")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, '风机数据分析报告.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"分析报告已保存至: {report_path}")
        
        return ''.join(report)
    
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("=" * 60)
        print("三台风机全量数据分析")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 清洗数据
        cleaning_stats = self.clean_data()
        
        # 3. 计算统计指标
        statistics = self.calculate_statistics()
        
        # 4. 检测异常
        anomalies = self.detect_anomalies()
        
        # 5. 生成可视化
        self.visualize_data()
        
        # 6. 生成报告
        report = self.generate_report(cleaning_stats, statistics, anomalies)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        print(f"结果已保存至目录: {self.output_dir}")
        print("包含内容:")
        print("  - 风机数据分析报告.md (分析报告)")
        print("  - 01-05 PNG图表文件 (可视化图表)")
        
        return {
            'cleaning_stats': cleaning_stats,
            'statistics': statistics,
            'anomalies': anomalies,
            'report': report
        }


if __name__ == '__main__':
    # 创建分析器实例
    analyzer = WindTurbineAnalyzer(data_dir='.')
    
    # 运行完整分析
    results = analyzer.run_complete_analysis()
    
    print("\n分析结果摘要:")
    print("-" * 60)
    if 'statistics' in results:
        for turbine_id, stats in results['statistics'].items():
            print(f"\n风机{turbine_id}:")
            print(f"  总发电量: {stats['总发电量(MWh)']:.2f} MWh")
            print(f"  平均功率: {stats['平均功率(kW)']:.2f} kW")
            print(f"  容量系数: {stats['容量系数(%)']:.2f}%")
            print(f"  可用率: {stats['可用率(%)']:.2f}%")
