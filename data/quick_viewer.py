import pandas as pd
import os

# 配置文件路径
input_file = '/home/myCourse/it5006/Team22_IT5006_Predictive_Policing_AY2526Sem2/data/exp_data/Chicago_Crimes_2015_2025.csv'
output_dir = '/home/myCourse/it5006/Team22_IT5006_Predictive_Policing_AY2526Sem2/data/quick-viewer'
output_file = os.path.join(output_dir, 'Chicago_Crimes_Quickview.csv')

# 创建目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("正在抽取每年样本数据...")

# 加载数据（由于已经过滤过，数据量减小，如果内存允许可直接读取）
# 如果还是很大，我们只读取需要的列
try:
    df = pd.read_csv(input_file, low_memory=False)
    
    # 按年份分组，每组取前100行
    quick_view_df = df.groupby('Year').head(100)
    
    # 按照年份和日期排序，方便查看
    quick_view_df = quick_view_df.sort_values(by=['Year', 'Date'], ascending=[False, False])
    
    # 保存结果
    quick_view_df.to_csv(output_file, index=False)
    
    print(f"抽样完成！预览文件已保存至: {output_file}")
    print(f"预览文件总行数: {len(quick_view_df)} 行")
    print(f"包含年份: {quick_view_df['Year'].unique()}")

except FileNotFoundError:
    print(f"错误：找不到文件 {input_file}，请先运行第一个脚本。")