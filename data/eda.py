import pandas as pd

# ====== 1. 检查 ID 是否有重复 ======
def check_id_duplicates(df):
    total_count = len(df)
    unique_count = df['ID'].nunique()
    duplicate_count = total_count - unique_count

    print("=== ID 检查 ===")
    print(f"总记录数: {total_count}")
    print(f"唯一 ID 数: {unique_count}")
    print(f"重复 ID 数: {duplicate_count}")

    if duplicate_count > 0:
        print("\n重复的 ID 示例：")
        print(df[df['ID'].duplicated()]['ID'].head())


# ====== 2. 检查缺失值 ======
def check_missing_values(df):
    print("\n=== 缺失值检查 ===")
    
    missing_community = df['Community Area'].isna().sum()
    missing_date = df['Date'].isna().sum()

    print(f"Community Area 缺失数: {missing_community}")
    print(f"Date 缺失数: {missing_date}")


# ====== 3. Primary Type 的 TOP K ======
def primary_type_topk(df, k=5):
    print(f"\n=== Primary Type TOP {k} ===")
    
    topk = df['Primary Type'].value_counts().head(k)
    print(topk)


# ====== 主函数调用 ======
def run_all_checks(df, k=5):
    check_id_duplicates(df)
    check_missing_values(df)
    primary_type_topk(df, k)


# ====== 示例调用 ======
df = pd.read_csv("/home/myCourse/it5006/Team22_IT5006_Predictive_Policing_AY2526Sem2/data/exp_data/Chicago_Crimes_2015_2025.csv")
run_all_checks(df, k=5)