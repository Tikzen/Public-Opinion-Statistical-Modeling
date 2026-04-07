import pandas as pd
from .base import BaseDataSource


class BaiduIndexSource(BaseDataSource):
    SOURCE_NAME = "baidu_index_file"

    def load_data(self, file_path: str) -> dict:
        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("仅支持 Excel 或 CSV 文件")

        date_col = None
        value_col = None

        for col in df.columns:
            if "时间" in col or "日期" in col:
                date_col = col
            if "搜索" in col and ("pc+移动" in col or "指数" in col):
                value_col = col

        if date_col is None or value_col is None:
            raise ValueError("未找到时间列或指数列")

        dates = df[date_col].astype(str).tolist()
        values = df[value_col].astype(float).tolist()

        keyword = None
        for col in df.columns:
            if "关键字" in col or "关键词" in col:
                keyword = str(df[col].iloc[0])
                break

        return {
            "dates": dates,
            "values": values,
            "keyword": keyword,
            "source": "baidu_index"
        }
