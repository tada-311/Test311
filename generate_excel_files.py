import pandas as pd
from pyproj import Transformer
import numpy as np
import os

# JGD2011平面直角座標系第10系 (EPSG:6670) の中心付近のWGS84座標
# 大分県庁の緯度経度を参考に、少し調整
center_lat_wgs84 = 33.23
center_lon_wgs84 = 131.61

# WGS84からJGD2011平面直角座標系第10系への変換器
transformer_to_jprcs10 = Transformer.from_crs("EPSG:4326", "EPSG:6670", always_xy=True)

# 中心座標をJPRCS第10系に変換 (pyprojのalways_xy=Trueは(easting, northing)順)
# JPRCSではXがNorthing, YがEastingなので、変換結果は(Y, X)の順で返る
center_easting_jprcs10, center_northing_jprcs10 = transformer_to_jprcs10.transform(center_lon_wgs84, center_lat_wgs84)

# 座標生成
num_points = 200

# X (Northing) と Y (Easting) の範囲を調整
# 中心座標から南北・東西にそれぞれ約10kmの範囲でランダムに生成
x_coords = np.random.uniform(center_northing_jprcs10 - 10000, center_northing_jprcs10 + 10000, num_points)
y_coords = np.random.uniform(center_easting_jprcs10 - 10000, center_easting_jprcs10 + 10000, num_points)
z_coords = np.random.uniform(0, 500, num_points) # 標高は0mから500mの範囲

# データフレームの作成
data = {'X': x_coords, 'Y': y_coords, 'Z': z_coords}
df = pd.DataFrame(data)

# ファイル保存パス
output_dir = r"C:\Users\KASEN001\Desktop\tada\Test"
os.makedirs(output_dir, exist_ok=True)

# 1. 横並び頂点.xlsx (ヘッダーが1行目、データが列方向)
output_file_horizontal = os.path.join(output_dir, "横並び頂点.xlsx")
df.to_excel(output_file_horizontal, index=False)
print(f"Created: {output_file_horizontal}")

# 2. 縦並び頂点.xlsx (ヘッダーが1列目、データが行方向)
# DataFrameを転置して保存
df_transposed = df.T
# ヘッダーを明示的に設定
df_transposed.index = ['X', 'Y', 'Z']
output_file_vertical = os.path.join(output_dir, "縦並び頂点.xlsx")
df_transposed.to_excel(output_file_vertical, header=False) # ヘッダー行は出力しない
print(f"Created: {output_file_vertical}")
