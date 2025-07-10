import streamlit as st
from pyproj import Transformer
from geopy.distance import geodesic
import numpy as np
import pandas as pd
import os
import re
import io

# --- 定数 ---
# 日本の緯度経度の範囲
japan_bounds = {
    "lat_min": 20.0,
    "lat_max": 46.0,
    "lon_min": 122.0,
    "lon_max": 154.0
}
# ジオイドモデルのファイルパス
GEOID_DATA_PATH = os.path.join(os.path.dirname(__file__), "gsigeo2011_ver2_2", "gsigeo2011_ver2_2.asc")

# --- データ読み込み・計算関数 ---

@st.cache_data
def load_geoid_data(file_path):
    """ジオイドデータファイルを読み込む"""
    if not os.path.exists(file_path):
        st.error(f"ジオイドデータファイルが見つかりません: {file_path}")
        return None, None, None, None, None

    with open(file_path, 'r') as f:
        header = f.readline().split()
        lat_start, lon_start, lat_interval, lon_interval = map(float, header[:4])
        num_lat, num_lon = map(int, header[4:])
        data = [float(val) for line in f for val in line.split()]
        geoid_heights = np.array(data).reshape(num_lat, num_lon)
    return geoid_heights, lat_start, lon_start, lat_interval, lon_interval

def get_geoid_height(lat, lon, geoid_heights, lat_start, lon_start, lat_interval, lon_interval):
    """緯度経度からジオイド高を計算（バイリニア補間）"""
    if geoid_heights is None: return None
    row = (lat - lat_start) / lat_interval
    col = (lon - lon_start) / lon_interval
    row_int, col_int = int(row), int(col)
    if not (0 <= row_int < geoid_heights.shape[0] - 1 and 0 <= col_int < geoid_heights.shape[1] - 1):
        return None
    h11, h12, h21, h22 = geoid_heights[row_int, col_int], geoid_heights[row_int, col_int + 1], geoid_heights[row_int + 1, col_int], geoid_heights[row_int + 1, col_int + 1]
    dx, dy = col - col_int, row - row_int
    geoid_h = (h11 * (1 - dx) * (1 - dy) + h12 * dx * (1 - dy) + h21 * (1 - dx) * dy + h22 * dx * dy)
    return geoid_h if geoid_h != 999.0000 else None

def auto_detect_zone(easting, northing):
    """座標値から最も可能性の高い系番号を自動判別する"""
    candidates = []
    for z_ in range(1, 20):
        try:
            transformer = Transformer.from_crs(f"EPSG:{6660 + z_}", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(easting, northing)
            if japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]:
                candidates.append({"zone": z_, "epsg": 6660 + z_, "lat": lat, "lon": lon})
        except Exception:
            continue
    if not candidates: return None
    reference_point = (33.23, 131.61)
    for c in candidates:
        c["distance"] = geodesic((c["lat"], c["lon"]), reference_point).meters
    best = min(candidates, key=lambda x: x["distance"])
    best["auto_detected"] = True
    return best

# --- 新しいファイル解析関数 ---
def parse_coordinate_file(uploaded_file):
    """アップロードされたExcel/CSVファイルを解析し、座標データを自動認識して抽出する"""
    if uploaded_file is None:
        return None, "ファイルがアップロードされていません。"
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == '.csv':
            content = uploaded_file.getvalue()
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')), header=None, dtype=str)
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(content.decode('shift-jis')), header=None, dtype=str)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file, header=None, engine='openpyxl', dtype=str)
        else:
            return None, "サポートされていないファイル形式です。Excel (.xlsx) または CSV (.csv) をアップロードしてください。"

        header_locs = {'x': None, 'y': None, 'z': None}
        found_headers = set()
        df_str = df.astype(str)

        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                cell_value = df_str.iat[r, c].lower()
                if not cell_value or cell_value == 'nan': continue
                
                if 'x' not in found_headers and re.search(r'x|easting', cell_value):
                    header_locs['x'] = (r, c); found_headers.add('x')
                elif 'y' not in found_headers and re.search(r'y|northing', cell_value):
                    header_locs['y'] = (r, c); found_headers.add('y')
                elif 'z' not in found_headers and re.search(r'z|height|標高', cell_value):
                    header_locs['z'] = (r, c); found_headers.add('z')

        if not all(header_locs.values()):
            missing = [k.upper() for k, v in header_locs.items() if v is None]
            return None, f"ヘッダーが見つかりません: {', '.join(missing)} を含むセルが必要です。"

        rows, cols = [loc[0] for loc in header_locs.values()], [loc[1] for loc in header_locs.values()]
        coordinates = []

        if len(set(rows)) == 1: # 横方向レイアウト
            header_row = rows[0]
            x_col, y_col, z_col = header_locs['x'][1], header_locs['y'][1], header_locs['z'][1]
            data_df = df.iloc[header_row + 1:]
            for _, row_data in data_df.iterrows():
                try:
                    coords = {'easting': float(row_data[x_col]), 'northing': float(row_data[y_col]), 'z': float(row_data[z_col])}
                    if not (pd.isna(coords['easting']) or pd.isna(coords['northing'])):
                        coordinates.append(coords)
                except (ValueError, TypeError, IndexError): continue
            return coordinates, None

        elif len(set(cols)) == 1: # 縦方向レイアウト
            header_col = cols[0]
            x_row, y_row, z_row = header_locs['x'][0], header_locs['y'][0], header_locs['z'][0]
            data_df = df.iloc[:, header_col + 1:]
            for col_idx in data_df.columns:
                try:
                    coords = {'easting': float(df.iat[x_row, col_idx]), 'northing': float(df.iat[y_row, col_idx]), 'z': float(df.iat[z_row, col_idx])}
                    if not (pd.isna(coords['easting']) or pd.isna(coords['northing'])):
                        coordinates.append(coords)
                except (ValueError, TypeError, IndexError): continue
            return coordinates, None
        else:
            return None, "ヘッダーのレイアウトを認識できません。X, Y, Z のヘッダーを同じ行または同じ列に揃えてください。"
    except Exception as e:
        return None, f"ファイルの処理中に予期せぬエラーが発生しました: {e}"

def parse_coordinate_text(input_string):
    """入力文字列から座標リストを抽出する"""
    coordinates = []
    pattern = r'[-+]?\d*\.?\d+'
    # 全ての数値を抽出
    numbers = re.findall(pattern, input_string)
    # 3つずつグループ化
    for i in range(0, len(numbers) - 2, 3):
        try:
            easting = float(numbers[i])
            northing = float(numbers[i+1])
            z = float(numbers[i+2])
            coordinates.append({'easting': easting, 'northing': northing, 'z': z})
        except ValueError:
            continue
    return coordinates

# --- Streamlit App ---
st.title("座標変換ツール (JGD2011平面直角座標系 → WGS84緯度経度)")

# パスワード認証
PASSWORD = os.environ.get("STREAMLIT_PASSWORD", "test")
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
    geoid_heights, lat_start, lon_start, lat_interval, lon_interval = load_geoid_data(GEOID_DATA_PATH)
    if geoid_heights is None:
        st.stop()

    st.markdown("---")
    st.subheader("座標入力")
    st.write("最大500個の座標を一括で変換できます。")

    input_method = st.radio("入力方法を選択:", ("ファイルアップロード", "テキスト入力"), horizontal=True)
    
    coordinate_input_text = ""
    uploaded_file = None

    if input_method == "ファイルアップロード":
        st.info(
            "**Excel (.xlsx) または CSV (.csv) ファイルをアップロードしてください。**\n\n"
            "ファイル内から `X`, `Y`, `Z` を含むヘッダーを探し、自動でデータを読み取ります。\n"
            "- **横方向データの場合:** ヘッダーを同じ **行** に配置 (例: A1='X', B1='Y', C1='Z')\n"
            "- **縦方向データの場合:** ヘッダーを同じ **列** に配置 (例: A1='X', A2='Y', A3='Z')\n"
        )
        uploaded_file = st.file_uploader("ファイルを選択", type=['xlsx', 'csv'])
    else:
        coordinate_input_text = st.text_area(
            '**X, Y, Z** の順で座標を入力してください。数値はスペース、カンマ、改行などで区切ってください。',
            height=250,
            value="""# ExcelからA,B,C列をコピー＆ペーストしても変換できます
# 例:
# 10000 20000 100
# -34567.89, 12345.67, 50.12
#
# -33696.311 13162.931 48.359
# -33692.211 13167.154 48.215
# -33687.894 13171.641 48.061
"""
        )

    col1, col2 = st.columns(2)
    with col1:
        zone_input = st.number_input('系番号 (自動判別は 0):', value=0, min_value=0, max_value=19)
    with col2:
        display_mode = st.radio("表示モード:", ("要約表示", "詳細表示"), horizontal=True)

    if st.button('変換実行', type="primary"):
        coordinates_to_convert = []
        if input_method == "ファイルアップロード":
            if uploaded_file:
                with st.spinner('ファイルを処理中...'):
                    coords, err = parse_coordinate_file(uploaded_file)
                if err:
                    st.error(f"⚠️ {err}")
                else:
                    coordinates_to_convert = coords
            else:
                st.warning("⚠️ ファイルが選択されていません。")
        else:
            coordinates_to_convert = parse_coordinate_text(coordinate_input_text)

        if not coordinates_to_convert:
            st.warning("⚠️ 変換対象の座標が見つかりませんでした。")
        elif len(coordinates_to_convert) > 500:
            st.error(f"⚠️ 一度に変換できる座標は最大500個です。(現在 {len(coordinates_to_convert)}個)")
        else:
            st.subheader("== 変換結果 (WGS84) ==")
            results_data = []
            progress_bar = st.progress(0)
            
            for i, coord in enumerate(coordinates_to_convert):
                easting, northing, z = coord['easting'], coord['northing'], coord['z']
                if easting == 0.0 and northing == 0.0: continue

                result_info = None
                if zone_input == 0:
                    result_info = auto_detect_zone(easting, northing)
                else:
                    try:
                        transformer = Transformer.from_crs(f"EPSG:{6660 + zone_input}", "EPSG:4326", always_xy=True)
                        lon, lat = transformer.transform(easting, northing)
                        if not (japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]):
                            result_info = auto_detect_zone(easting, northing)
                            if result_info: result_info["manual_fail"] = True
                        else:
                            result_info = {"zone": zone_input, "epsg": 6660 + zone_input, "lat": lat, "lon": lon, "auto_detected": False}
                    except Exception:
                        result_info = None
                
                ellipsoidal_height, geoid_h = None, None
                if result_info:
                    geoid_h = get_geoid_height(result_info['lat'], result_info['lon'], geoid_heights, lat_start, lon_start, lat_interval, lon_interval)
                    if geoid_h is not None:
                        ellipsoidal_height = z + geoid_h

                results_data.append({"id": i + 1, "input": coord, "result": result_info, "geoid_height": geoid_h, "ellipsoidal_height": ellipsoidal_height})
                progress_bar.progress((i + 1) / len(coordinates_to_convert))

            if display_mode == "詳細表示":
                for res in results_data:
                    st.markdown(f"--- \n### **座標 {res['id']}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**入力座標 (平面直角座標系)**")
                        st.write(f"X: `{res['input']['easting']}` | Y: `{res['input']['northing']}` | Z: `{res['input']['z']}`")
                    with col2:
                        st.write("**変換結果 (WGS84)**")
                        if res["result"]:
                            st.write(f"緯度: `{res['result']['lat']:.8f}` | 経度: `{res['result']['lon']:.8f}`")
                            if res["ellipsoidal_height"] is not None:
                                st.write(f"楕円体高: `{res['ellipsoidal_height']:.4f} m` (ジオイド高: `{res['geoid_height']:.4f} m`)")
                            else:
                                st.warning("ジオイド高取得不可")
                            zone_str = f"第{res['result']['zone']}系"
                            if res['result'].get("auto_detected"): zone_str += " (自動判別)"
                            if res['result'].get("manual_fail"): zone_str += f" (指定の{zone_input}系は範囲外のため自動判別)"
                            st.write(f"系番号: `{zone_str}`")
                        else:
                            st.error("座標変換失敗")
            else: # 要約表示
                summary_data = []
                for res in results_data:
                    if res["result"]:
                        h_str = f"{res['ellipsoidal_height']:.3f}" if res['ellipsoidal_height'] is not None else "N/A"
                        zone_str = f"{res['result']['zone']}"
                        if res['result'].get("auto_detected"): zone_str += "*"
                        summary_data.append({"点": res["id"], "緯度": f"{res['result']['lat']:.7f}", "経度": f"{res['result']['lon']:.7f}", "楕円体高(m)": h_str, "系": zone_str})
                    else:
                        summary_data.append({"点": res["id"], "緯度": "変換失敗", "経度": "", "楕円体高(m)": "", "系": ""})
                st.dataframe(summary_data, use_container_width=True)
                st.caption("* が付いている系番号は自動判別されたものです。")