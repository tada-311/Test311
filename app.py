import streamlit as st
from pyproj import Transformer
from geopy.distance import geodesic
import os
import re
import io
import pandas as pd # pandasを再インポート

# --- 定数 ---
japan_bounds = {
    "lat_min": 20.0,
    "lat_max": 46.0,
    "lon_min": 122.0,
    "lon_max": 154.0
}

# --- 緯度経度変換ヘルパー ---
def decimal_to_dms_string(decimal_degrees):
    """10進数の度を度分秒形式の文字列に変換 (例: 370856.12340)"""
    if decimal_degrees is None: # Noneの場合のハンドリングを追加
        return ""

    degrees = int(decimal_degrees)
    minutes_decimal = (decimal_degrees - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60

    # 秒を小数点以下5桁までフォーマット
    seconds_str = f"{seconds:.5f}"
    # 秒の整数部分が1桁の場合に0埋め
    if seconds < 10 and seconds >= 0: # 0.00000-9.99999 の場合
        seconds_str = f"0{seconds_str}"
    elif seconds < 0 and seconds > -10: # -0.00000 - -9.99999 の場合
        seconds_str = f"-0{seconds_str[1:]}" # 符号はそのままに0埋め

    # 分を2桁で0埋め
    minutes_str = f"{minutes:02d}"

    # 結合して指定された形式の文字列を生成
    # 例: 370856.12340
    return f"{degrees}{minutes_str}{seconds_str}"

# --- 座標変換ヘルパー ---
def auto_detect_zone(easting, northing):
    candidates = []
    for z_ in range(1, 20):
        epsg_code = 6660 + z_
        try:
            transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(easting, northing)
            if (
                japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and
                japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]
            ):
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
    # ここでは仮に大分県の座標を基準点とする
    reference_point = (33.23, 131.61) 
    for c in candidates:
        c["distance"] = geodesic((c["lat"], c["lon"]), reference_point).meters
    best = min(candidates, key=lambda x: x["distance"])
    best["auto_detected"] = True
    return best

# --- ファイル解析関数 ---
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
                    except:
                        # データが数値でない場合はブロックの終わりと判断
                        break
                if block_coords:
                    all_coords.extend(block_coords)
                    used.update({(r_x, c_x), y_match})

        return all_coords, None if all_coords else "X, Yのヘッダーが見つかりませんでした。"
    except Exception as e:
        return None, f"エラー: {e}"

def parse_coordinate_text(input_string):
    coordinates = []
    # 1行ずつ処理
    for line_num, line in enumerate(input_string.splitlines(), 1):
        # 行から数値を抽出
        found_numbers = re.findall(r'[-+]?\d*\.?\d+', line)
        
        if len(found_numbers) == 2:
            # X, Y の順で解釈 (Easting, Northing)
            easting = float(found_numbers[0])
            northing = float(found_numbers[1])
            coordinates.append({'easting': easting, 'northing': northing})
        elif len(found_numbers) > 2:
            easting = float(found_numbers[0])
            northing = float(found_numbers[1])
            coordinates.append({'easting': easting, 'northing': northing})
            st.warning(f"⚠️ {line_num}行目: 2つより多くの数値が見つかりました。最初の2つ ({easting}, {northing}) を使用します。")
        elif len(found_numbers) > 0:
            st.warning(f"⚠️ {line_num}行目: 座標の数値が2つではありません。スキップされます。: `{line}`")

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
        coordinate_input_text = st.text_area(
            '**X, Y** の順で座標を入力してください。\n\n'
            '1行に1座標ずつ入力します。数値はスペース、カンマ、タブなどで区切ってください。\n\n'
            '例:\n'
            '`-36258.580  -147524.100`\n'
            '`X=-36258.580, Y=-147524.100`',
            height=150
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
            st.error(f"⚠️ 最大500個まで。(現在 {len(coordinates_to_convert)}個)")
        else:
            st.subheader("== 変換結果 (WGS84) ==")
            results_data = []
            progress_bar = st.progress(0)
            
            for i, coord in enumerate(coordinates_to_convert):
                easting, northing = coord['easting'], coord['northing']
                if easting == 0.0 and northing == 0.0:
                    st.warning(f"⚠️ **点 {i+1}**: 座標値が (0, 0) のためスキップしました。")
                    continue

                result_info = None
                if zone_input == 0:
                    result_info = auto_detect_zone(easting, northing)
                else:
                    try:
                        transformer = Transformer.from_crs(f"EPSG:{6660 + zone_input}", "EPSG:4326", always_xy=True)
                        lon, lat = transformer.transform(easting, northing)
                        if not (japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]):
                            st.warning(f"⚠️ **点 {i+1}**: 指定された第{zone_input}系で変換すると日本の範囲外になります。自動判別を試します。")
                            result_info = auto_detect_zone(easting, northing)
                        else:
                            result_info = {"zone": zone_input, "epsg": 6660 + zone_input, "lat": lat, "lon": lon, "auto_detected": False}
                    except Exception as e:
                        st.error(f"⚠️ **点 {i+1}**: 第{zone_input}系での変換中にエラーが発生しました: {e}")
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
                        if res['result'].get("auto_detected"):
                            zone_str += " (自動判別)"
                        st.write(f"系番号: `{zone_str}`")
                    else:
                        st.error("座標変換失敗")
            else:
                summary_data = []
                for res in results_data:
                    if res["result"]:
                        zone_str = f"{res['result']['zone']}"
                        if res['result'].get("auto_detected"):
                            zone_str += "*" # 自動判別は*マーク
                        summary_data.append({
                            "点": res["id"],
                            "緯度": f"{res['result']['lat']:.10f}",
                            "経度": f"{res['result']['lon']:.10f}",
                            "系": zone_str
                        })
                    else:
                        summary_data.append({"点": res["id"], "緯度": "変換失敗", "経度": "", "系": ""})
                st.dataframe(summary_data, use_container_width=True)
                st.caption("* が付いている系番号は自動判別されたものです。")

            # ジオイド高計算用ファイル出力ボタン
            geoid_in_content = "# 緯度(dms)   経度(dms)\n"
            for res in results_data:
                if res["result"]:
                    lat_dms = decimal_to_dms_string(res["result"]["lat"])
                    lon_dms = decimal_to_dms_string(res["result"]["lon"])
                    geoid_in_content += f"{lat_dms} {lon_dms}\n"
            
            if geoid_in_content != "# 緯度(dms)   経度(dms)\n": # ヘッダー行以外にデータがある場合のみボタンを表示
                st.download_button(
                    label="ジオイド高計算用ファイル (.in) をダウンロード",
                    data=geoid_in_content.encode('shift-jis'), # Shift-JISでエンコード
                    file_name="geoid.in",
                    mime="text/plain"
                )