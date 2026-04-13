import pandas as pd
import os

# 配置文件路径
input_file = '/home/myCourse/it5006/Team22_IT5006_Predictive_Policing_AY2526Sem2/data/origin/Crimes_-_2001_to_Present_20260216.csv'  # 原始文件名
output_dir = '/home/myCourse/it5006/Team22_IT5006_Predictive_Policing_AY2526Sem2/data/exp_data'
output_file = os.path.join(output_dir, 'Chicago_Crimes_2015_2025.csv')

# 创建目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建目录: {output_dir}")

print("正在开始过滤数据，请稍候...")

# 使用分块读取，防止内存溢出（每块10万行）
chunk_size = 100000
first_chunk = True

# 计数器
total_rows = 0

for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False):
    # 过滤条件：Year 字段在 2015 到 2025 之间
    # 注意：Chicago 数据集自带 'Year' 列，无需手动解析日期，速度极快
    filtered_chunk = chunk[(chunk['Year'] >= 2015) & (chunk['Year'] <= 2025)]
    
    if not filtered_chunk.empty:
        # 写入新文件
        # 如果是第一块，则写入表头(header=True)，否则追加(mode='a', header=False)
        filtered_chunk.to_csv(output_file, mode='a', index=False, header=first_chunk)
        first_chunk = False
        total_rows += len(filtered_chunk)
        print(f"已处理并保存 {total_rows} 行数据...", end='\r')

print(f"\n处理完成！新文件已保存至: {output_file}")
print(f"总计保留行数: {total_rows}")