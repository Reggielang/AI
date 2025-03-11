from datetime import datetime, date

import pandas as pd

# 定义节假日列表（可以根据需要扩展）
holidays = [
    "2025-01-01",  # 元旦
    "2025-01-28",  # 春节
    "2025-01-29",
    "2025-01-30",
    "2025-01-31",
    "2025-02-01",
    "2025-02-02",
    "2025-02-03",
    "2025-02-04",
    "2025-04-05",  # 清明节
    "2025-05-01",  # 劳动节
    "2025-06-22",  # 端午节
    "2025-09-29",  # 中秋节
    "2025-10-01",  # 国庆节
    "2025-10-02",
    "2025-10-03",
    "2025-10-04",
    "2025-10-05",
    "2025-10-06",
]


def check_day_type(target_date=None):
    """
    判断给定日期是节假日、周末还是工作日。

    参数:
        target_date (str or date): 目标日期，格式为 "YYYY-MM-DD" 或 datetime.date 对象。
                                  如果为 None，则使用当天日期。

    返回:
        str: "节假日"、"周末" 或 "工作日"
    """
    # 如果没有提供日期，则使用当天日期
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    # 判断是否是周末
    if target_date.weekday() >= 5:  # 5: 周六, 6: 周日
        return "周末"

    # 判断是否是节假日
    if target_date.strftime("%Y-%m-%d") in holidays:
        return "节假日"

    # 默认返回工作日
    return "工作日"


# 示例用法
if __name__ == "__main__":
    df1 = pd.read_excel(r'C:\Users\KODI\Desktop\Workbook3.xlsx',sheet_name='日期判断')
    for index, row in df1.iterrows():
        print(row[0],check_day_type(row[0]))
        # print(row,check_day_type(row))