import streamlit as st
from pyproj import Transformer
import pandas as pd
import os
import re
import io

# --- 定数 ---
japan_bounds = {
    "lat_min": 20.0,
    "lat_max": 46.0,
    "lon_min": 122.0,
    "lon_max": 154.0
}

# --- 座標変換ヘルパー ---
def auto_detect_zone(easting, northing):
    for zone in range(1, 20):
        try:
            transformer = Transformer.from_crs(f"EPSG:{6660 + zone}", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(northing, easting)
            if japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]:
                return {"zone": zone, "epsg": 6660 + zone, "lat": lat, "lon": lon, "auto_detected": True}
        except Exception:
            continue
    return None

# --- ファイル解析関数（Z削除済） ---
def parse_coordinate_file(uploaded_file):
    if uploaded_file is None:
        return None, "ファイルがアップロードされていません。"
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == '.csv':
            content = uploaded_file.getvalue()
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')), header=None, dtype=str)
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(content.decode('shift-jis')), header=None, dtype=str)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file, header=None, engine='openpyxl', dtype=str)
        else:
            return None, "サポートされていないファイル形式です。"

        all_coords = []
        df_str = df.astype(str)

        x_locs, y_locs = [], []

        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                val = df_str.iat[r, c].lower()
                if not val or val == 'nan': continue
                if re.search(r'x|easting', val): x_locs.append((r, c))
                elif re.search(r'y|northing', val): y_locs.append((r, c))

        used = set()

        for r_x, c_x in x_locs:
            if (r_x, c_x) in used: continue
            y_match = next(((r_y, c_y) for r_y, c_y in y_locs if r_y == r_x and (r_y, c_y) not in used), None)

            if y_match:
                header_row = r_x
                x_col, y_col = c_x, y_match[1]
                block_coords = []
                for r_data in range(header_row + 1, df.shape[0]):
                    try:
                        easting = float(df.iat[r_data, x_col])
                        northing = float(df.iat[r_data, y_col])
                        block_coords.append({'easting': easting, 'northing': northing})
                    except: break
                if block_coords:
                    all_coords.extend(block_coords)
                    used.update({(r_x, c_x), y_match})

        return all_coords, None if all_coords else "X, Yのヘッダーが見つかりませんでした。"
    except Exception as e:
        return None, f"エラー: {e}"

def parse_coordinate_text(input_string):
    coordinates = []
    pattern = r'[-+]?\d*\.?\d+'
    numbers = re.findall(pattern, input_string)
    for i in range(0, len(numbers) - 1, 2):
        try:
            easting = float(numbers[i])
            northing = float(numbers[i+1])
            coordinates.append({'easting': easting, 'northing': northing})
        except: continue
    return coordinates

# --- Streamlit App ---
st.title("座標変換ツール (JGD2011平面直角座標系 → WGS84緯度経度)")

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
    st.subheader("座標入力")
    input_method = st.radio("入力方法を選択:", ("ファイルアップロード", "テキスト入力"), horizontal=True)
    
    coordinate_input_text = ""
    uploaded_file = None

    if input_method == "ファイルアップロード":
        st.info("Excel (.xlsx) または CSV (.csv) ファイルをアップロードしてください。")
        uploaded_file = st.file_uploader("ファイルを選択", type=['xlsx', 'csv'])
    else:
        coordinate_input_text = st.text_area('X, Y の順で座標を入力してください。', height=250, value="")

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
            st.error(f"⚠️ 最大500個まで。(現在 {len(coordinates_to_convert)}個)")
        else:
            st.subheader("== 変換結果 (WGS84) ==")
            results_data = []
            progress_bar = st.progress(0)
            
            for i, coord in enumerate(coordinates_to_convert):
                easting, northing = coord['easting'], coord['northing']
                if easting == 0.0 and northing == 0.0: continue

                result_info = None
                if zone_input == 0:
                    result_info = auto_detect_zone(easting, northing)
                else:
                    try:
                        transformer = Transformer.from_crs(f"EPSG:{6660 + zone_input}", "EPSG:4326", always_xy=True)
                        lon, lat = transformer.transform(northing, easting)
                        if not (japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]):
                            result_info = auto_detect_zone(easting, northing)
                            if result_info: result_info["manual_fail"] = True
                        else:
                            result_info = {"zone": zone_input, "epsg": 6660 + zone_input, "lat": lat, "lon": lon, "auto_detected": False}
                    except:
                        result_info = None

                results_data.append({"id": i + 1, "input": coord, "result": result_info})
                progress_bar.progress((i + 1) / len(coordinates_to_convert))

            if display_mode == "詳細表示":
                for res in results_data:
                    st.markdown(f"--- \n### **座標 {res['id']}**")
                    st.write(f"**X:** {res['input']['easting']} | **Y:** {res['input']['northing']}")
                    if res["result"]:
                        st.write(f"緯度: `{res['result']['lat']:.10f}` | 経度: `{res['result']['lon']:.10f}`")
                        zone_str = f"{res['result']['zone']}"
                        if res['result'].get("auto_detected"): zone_str += " (自動判別)"
                        if res['result'].get("manual_fail"): zone_str += " (指定系は範囲外)"
                        st.write(f"系番号: `{zone_str}`")
                    else:
                        st.error("座標変換失敗")
            else:
                summary_data = []
                for res in results_data:
                    if res["result"]:
                        zone_str = f"{res['result']['zone']}"
                        if res['result'].get("auto_detected"): zone_str += "*"
                        summary_data.append({
                            "点": res["id"],
                            "緯度": f"{res['result']['lat']:.10f}",
                            "経度": f"{res['result']['lon']:.10f}",
                            "系": zone_str
                        })
                    else:
                        summary_data.append({"点": res["id"], "緯度": "変換失敗", "経度": "", "系": ""})
                st.dataframe(summary_data, use_container_width=True)
