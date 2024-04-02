import pandas as pd
import zipfile
import os
from datetime import datetime
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF

def download_and_process(start_year, start_month, pair='eurusd', platform=P.NINJA_TRADER, time_frame=TF.ONE_MINUTE):
    current_year = datetime.now().year
    current_month = datetime.now().month

    all_data = []

    for year in range(start_year, current_year + 1):
        if year < current_year:
            # 对于起始年份到当前年份之前的每一年，下载整年数据
            path = download_data(year, None, pair, platform, time_frame)
            process_and_append_data(path, all_data)
        else:
            # 对于当前年份，下载每个月的数据直到当前月份
            for month in range(1, current_month ):
                path = download_data(year, month, pair, platform, time_frame)
                process_and_append_data(path, all_data)

    # 组合所有数据到一个 DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def download_data(year, month, pair, platform, time_frame):
    if month is not None:
        return dl(year=str(year), month=str(month).zfill(2), pair=pair, platform=platform, time_frame=time_frame)
    else:
        return dl(year=str(year), pair=pair, platform=platform, time_frame=time_frame, month=None)

def process_and_append_data(path, all_data):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall("extracted_data")

    extracted_file_name = os.listdir("extracted_data")[0]
    extracted_file_path = os.path.join("extracted_data", extracted_file_name)

    data = pd.read_csv(extracted_file_path, delimiter=';', names=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    all_data.append(data)

    # 清理解压缩的文件以节省空间
    os.remove(extracted_file_path)

# 示例使用
try:
    data = download_and_process(2021, 7)
    data.to_csv("data.csv", index=False)
    print("数据已成功下载和处理。")
except Exception as e:
    print(f"发生错误: {e}")
