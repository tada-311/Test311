import streamlit as st
from pyproj import Transformer
from geopy.distance import geodesic
import numpy as np
import os
import re # For more robust parsing

# 日本の緯度経度範囲
japan_bounds = {
    "lat_min": 20.0,
    "lat_max": 46.0,
    "lon_min": 122.0,
    "lon_max": 154.0
}

# ジオイドモデルデータのパス
GEOID_DATA_PATH = "./gsigeo2011_ver2_2/gsigeo2011_ver2_2.asc"

# ジオイド高データを読み込む関数
@st.cache_data
def load_geoid_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"ジオイドデータファイルが見つかりません: {file_path}")
        return None, None, None, None, None

    with open(file_path, 'r') as f:
        header = f.readline().split()
        # ヘッダ情報の解析
        # 20.00000 120.00000 0.016667 0.025000 1801 1201 1 ver2.2
        lat_start = float(header[0])
        lon_start = float(header[1])
        lat_interval = float(header[2])
        lon_interval = float(header[3])
        num_lat = int(header[4])
        num_lon = int(header[5])

        # ジオイド高データの読み込み
        data = []
        for line in f:
            data.extend(map(float, line.split()))
        
        # numpy配列に変換し、形状を調整
        geoid_heights = np.array(data).reshape(num_lat, num_lon)
        
        return geoid_heights, lat_start, lon_start, lat_interval, lon_interval

# ジオイド高を補間して取得する関数
def get_geoid_height(lat, lon, geoid_heights, lat_start, lon_start, lat_interval, lon_interval):
    if geoid_heights is None:
        return None

    # グリッド座標に変換
    row = (lat - lat_start) / lat_interval
    col = (lon - lon_start) / lon_interval

    # 補間
    row_int = int(row)
    col_int = int(col)
    
    # 範囲チェック
    if not (0 <= row_int < geoid_heights.shape[0] - 1 and 0 <= col_int < geoid_heights.shape[1] - 1):
        return None

    # 4点補間
    h11 = geoid_heights[row_int, col_int]
    h12 = geoid_heights[row_int, col_int + 1]
    h21 = geoid_heights[row_int + 1, col_int]
    h22 = geoid_heights[row_int + 1, col_int + 1]

    dx = col - col_int
    dy = row - row_int

    geoid_h = (h11 * (1 - dx) * (1 - dy) +
               h12 * dx * (1 - dy) +
               h21 * (1 - dx) * dy +
               h22 * dx * dy)
    
    # 999.0000 は欠損値なので、その場合はNoneを返す
    if geoid_h == 999.0000:
        return None
    
    return geoid_h

def auto_detect_zone(easting, northing):
    candidates = []
    for z_ in range(1, 20):
        epsg_code = 6660 + z_
        try:
            transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(easting, northing)
            if (japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and
                japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]):
                candidates.append({
                    "zone": z_,
                    "epsg": epsg_code,
                    "lat": lat,
                    "lon": lon
                })
        except Exception:
            continue
    if not candidates:
        return None
    reference_point = (33.5, 131.0) # 日本の中心に近い参照点
    for c in candidates:
        c["distance"] = geodesic((c["lat"], c["lon"]), reference_point).meters
    best = min(candidates, key=lambda x: x["distance"])
    best["auto_detected"] = True
    return best

# --- 座標解析関数 ---
def parse_coordinate_input(input_string):
    coordinates = []
    # すべての行から数値を抽出
    numbers = []
    # 数字（小数点、符号含む）を抽出する正規表現。単位やラベルを無視する。
    # re.findall はマッチしたすべての部分文字列をリストで返す
    # input_string.splitlines() で各行を処理し、より柔軟な入力に対応
    for line in input_string.splitlines():
        found_numbers = re.findall(r'[-+]?\d*\.?\d+', line)
        for num_str in found_numbers:
            try:
                numbers.append(float(num_str))
            except ValueError:
                # 数値に変換できない場合はスキップ
                pass

    # 抽出された数値を3つずつグループ化して座標とする
    for i in range(0, len(numbers), 3):
        if i + 2 < len(numbers): # X, Y, Z の3つが揃っているか確認
            northing = numbers[i]
            easting = numbers[i+1]
            z = numbers[i+2]
            coordinates.append({'northing': northing, 'easting': easting, 'z': z})
        else:
            if len(numbers) > 0: # 数値が全くない場合は警告しない
                st.warning("⚠️ 入力された数値の数が不完全な座標セットを検出しました。スキップされます。")
            break # 不完全なセットがあればそこで終了

    return coordinates

st.title("座標変換ツール")

# パスワード設定 (実際の運用ではst.secretsを使用することを推奨)
PASSWORD = "test"

if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

if not st.session_state["password_correct"]:
    password_input = st.text_input("パスワードを入力してください", type="password")
    if st.button("ログイン"):
        if password_input == PASSWORD:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("パスワードが間違っています")
else:
    # ジオイドデータをロード
    geoid_heights, lat_start, lon_start, lat_interval, lon_interval = load_geoid_data(GEOID_DATA_PATH)

    if geoid_heights is None:
        st.stop() # データロードに失敗したら処理を停止

    st.markdown("--- ")
    st.subheader("座標入力")
    st.write("複数の座標を入力できます。")
    st.write("例: `X1,Y1,Z1/X2,Y2,Z2` またはExcelからのコピー＆ペースト（縦並び、横並び、単位の有無に対応）")
        st.write("最大500個の座標まで対応。")

    coordinate_input_text = st.text_area(
        'X, Y, Z 座標を入力してください (各座標はカンマ、スペース、タブ、改行で区切る。単位やラベルは自動で無視されます)',
        height=150,
        help="例:\nX 12345.67 (m)\nY 67890.12 (m)\nZ 100.0 (m)\n\nまたは\n\n12345.67, 67890.12, 100.0\n12345.67 67890.12 100.0\n12345.67\t67890.12\t100.0\n\nまたは\n\n12345.67,67890.12,100.0/12345.67,67890.12,100.0"
    )

    zone_input = st.number_input('系番号 (1〜19で指定、自動判別は0):', value=0, min_value=0, max_value=19, step=1)

    display_mode = st.radio("表示モードを選択してください:", ("要約表示", "詳細表示"))

    if st.button('変換実行'):
        coordinates_to_convert = parse_coordinate_input(coordinate_input_text)

        if not coordinates_to_convert:
            st.warning("⚠️ 変換する座標が入力されていません。")
        elif len(coordinates_to_convert) > 500:
            st.warning(f"⚠️ 入力された座標の数が多すぎます ({len(coordinates_to_convert)}個)。最大500個までです。")
        else:
            st.subheader("=== 座標変換結果（WGS84） ===")
            for i, coord in enumerate(coordinates_to_convert):
                northing = coord['northing']
                easting = coord['easting']
                z = coord['z']

                if easting == 0.0 or northing == 0.0:
                    st.warning(f"⚠️ 座標 {i+1}: Y座標（東ing）とX座標（北ing）は0以外を入力してください。この座標はスキップされます。")
                    continue

                result = None
                if zone_input == 0:
                    result = auto_detect_zone(easting, northing)
                elif 1 <= zone_input <= 19:
                    try:
                        epsg_code = 6660 + zone_input
                        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
                        lon, lat = transformer.transform(easting, northing)
                        if (japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and
                            japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]):
                            result = {
                                "zone": zone_input,
                                "epsg": epsg_code,
                                "lat": lat,
                                "lon": lon,
                                "auto_detected": False
                            }
                    except Exception:
                        result = None

                if display_mode == "詳細表示":
                    st.markdown(f"--- **座標 {i+1}** ---")
                    st.write(f"入力: X={northing}, Y={easting}, Z={z}")

                    if result:
                        ellipsoidal_height = None # Initialize ellipsoidal_height
                        # ジオイド高を自動取得
                        auto_geoid_height = get_geoid_height(result['lat'], result['lon'], geoid_heights, lat_start, lon_start, lat_interval, lon_interval)

                        if auto_geoid_height is not None:
                            ellipsoidal_height = z + auto_geoid_height
                            st.write(f"系番号: 第{result['zone']}系")
                            st.write(f"EPSGコード: {result['epsg']}")
                            st.write(f"緯度（北緯）: {result['lat']:.15f}")
                            st.write(f"経度（東経）: {result['lon']:.15f}")
                            st.write(f"自動取得ジオイド高: {auto_geoid_height:.15f} m")
                            st.write(f"楕円体高（Z + 自動取得ジオイド高）: {ellipsoidal_height:.15f} m")
                            if result.get("auto_detected", False):
                                st.info("※ 系番号は自動判別されました。")
                        else:
                            st.warning("⚠️ ジオイド高の自動取得に失敗しました。座標がジオイドモデルの範囲内か確認してください。")
                            st.write(f"系番号: 第{result['zone']}系")
                            st.write(f"EPSGコード: {result['epsg']}")
                            st.write(f"緯度（北緯）: {result['lat']:.15f}")
                            st.write(f"経度（東経）: {result['lon']:.15f}")
                            st.warning("ジオイド高が自動取得できなかったため、楕円体高は計算できませんでした。")

                    else:
                        st.error("⚠️ 有効な座標変換結果が得られませんでした。座標または系番号が正しいか確認してください。")
                else: # 要約表示
                    if result:
                        ellipsoidal_height_str = f"{ellipsoidal_height:.2f}" if ellipsoidal_height is not None else "N/A"
                        st.write(f"点{i+1}: 入力 X={northing:.2f}, Y={easting:.2f}, Z={z:.2f} -> 緯度={result['lat']:.6f}, 経度={result['lon']:.6f}, 楕円体高={ellipsoidal_height_str} m")
                    else:
                        st.warning(f"点{i+1}: 変換失敗 - 入力 X={northing:.2f}, Y={easting:.2f}, Z={z:.2f}")