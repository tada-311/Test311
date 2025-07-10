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
        num_lat = int(header[4])
        num_lon = int(header[5])
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
    reference_point = (33.5, 131.0)
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

        all_coordinates = []
        df_str = df.astype(str)

        # すべてのX, Y, Zヘッダーの位置を検出
        x_locs = []
        y_locs = []
        z_locs = []

        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                cell_value = df_str.iat[r, c].lower()
                if not cell_value or cell_value == 'nan': continue
                
                if re.search(r'x|easting', cell_value):
                    x_locs.append((r, c))
                elif re.search(r'y|northing', cell_value):
                    y_locs.append((r, c))
                elif re.search(r'z|height|標高', cell_value):
                    z_locs.append((r, c))

        used_header_cells = set()
        
        # 横方向のヘッダーブロックを探索
        for r_x, c_x in x_locs:
            if (r_x, c_x) in used_header_cells: continue
            
            # 同じ行でYとZを探す
            y_match = None
            for r_y, c_y in y_locs:
                if r_y == r_x and (r_y, c_y) not in used_header_cells:
                    y_match = (r_y, c_y)
                    break
            
            z_match = None
            if y_match:
                for r_z, c_z in z_locs:
                    if r_z == r_x and (r_z, c_z) not in used_header_cells:
                        z_match = (r_z, c_z)
                        break
            
            if y_match and z_match:
                # 有効な横方向ヘッダーブロックが見つかった
                header_row = r_x
                x_col, y_col, z_col = c_x, y_match[1], z_match[1]
                
                # データの抽出
                current_block_coords = []
                for r_data in range(header_row + 1, df.shape[0]):
                    try:
                        easting = float(df.iat[r_data, x_col])
                        northing = float(df.iat[r_data, y_col])
                        z = float(df.iat[r_data, z_col])
                        if not (pd.isna(easting) or pd.isna(northing) or pd.isna(z)):
                            current_block_coords.append({'easting': easting, 'northing': northing, 'z': z})
                    except (ValueError, TypeError, IndexError):
                        # データが数値でない、または範囲外になったらこのブロックの処理を終了
                        break
                
                if current_block_coords:
                    all_coordinates.extend(current_block_coords)
                    used_header_cells.add((r_x, c_x))
                    used_header_cells.add(y_match)
                    used_header_cells.add(z_match)

        # 縦方向のヘッダーブロックを探索
        for r_x, c_x in x_locs:
            if (r_x, c_x) in used_header_cells: continue # 既に横方向で使われていたらスキップ
            
            # 同じ列でYとZを探す
            y_match = None
            for r_y, c_y in y_locs:
                if c_y == c_x and (r_y, c_y) not in used_header_cells:
                    y_match = (r_y, c_y)
                    break
            
            z_match = None
            if y_match:
                for r_z, c_z in z_locs:
                    if c_z == c_x and (r_z, c_z) not in used_header_cells:
                        z_match = (r_z, c_z)
                        break
            
            if y_match and z_match:
                # 有効な縦方向ヘッダーブロックが見つかった
                header_col = c_x
                x_row, y_row, z_row = r_x, y_match[0], z_match[0]
                
                # データの抽出
                current_block_coords = []
                for c_data in range(header_col + 1, df.shape[1]):
                    try:
                        easting = float(df.iat[x_row, c_data])
                        northing = float(df.iat[y_row, c_data])
                        z = float(df.iat[z_row, c_data])
                        if not (pd.isna(easting) or pd.isna(northing) or pd.isna(z)):
                            current_block_coords.append({'easting': easting, 'northing': northing, 'z': z})
                    except (ValueError, TypeError, IndexError):
                        # データが数値でない、または範囲外になったらこのブロックの処理を終了
                        break
                
                if current_block_coords:
                    all_coordinates.extend(current_block_coords)
                    used_header_cells.add((r_x, c_x))
                    used_header_cells.add(y_match)
                    used_header_cells.add(z_match)

        if not all_coordinates:
            return None, "ファイルから有効な座標データが見つかりませんでした。X, Y, Z のヘッダーを同じ行または同じ列に揃えてください。"
        
        return all_coordinates, None

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
            value=""
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
                st.caption("アスタリスク (*) が付いている系番号は自動判別されたものです。")