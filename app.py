import streamlit as st
from pyproj import Transformer
from geopy.distance import geodesic
import numpy as np
import os
import re

# 日本の緯度経度の範囲
japan_bounds = {
    "lat_min": 20.0,
    "lat_max": 46.0,
    "lon_min": 122.0,
    "lon_max": 154.0
}

# ジオイドモデルのファイルパス
# このスクリプト(app.py)と同じ階層にgsigeo2011_ver2_2フォルダを置いてください
GEOID_DATA_PATH = os.path.join(os.path.dirname(__file__), "gsigeo2011_ver2_2", "gsigeo2011_ver2_2.asc")

@st.cache_data
def load_geoid_data(file_path):
    """ジオイドデータファイルを読み込む"""
    if not os.path.exists(file_path):
        # ファイルが見つからない場合、カレントディレクトリからの相対パスも試す
        alt_path = "./gsigeo2011_ver2_2/gsigeo2011_ver2_2.asc"
        if not os.path.exists(alt_path):
            st.error(f"ジオイドデータファイルが見つかりません: {file_path} または {alt_path}")
            return None, None, None, None, None
        file_path = alt_path

    with open(file_path, 'r') as f:
        # ヘッダー情報の読み込み
        header = f.readline().split()
        lat_start = float(header[0])
        lon_start = float(header[1])
        lat_interval = float(header[2])
        lon_interval = float(header[3])
        num_lat = int(header[4])
        num_lon = int(header[5])

        data = []
        for line in f:
            data.extend(map(float, line.split()))

        geoid_heights = np.array(data).reshape(num_lat, num_lon)

    return geoid_heights, lat_start, lon_start, lat_interval, lon_interval

def get_geoid_height(lat, lon, geoid_heights, lat_start, lon_start, lat_interval, lon_interval):
    """緯度経度からジオイド高を計算（バイリニア補間）"""
    if geoid_heights is None:
        return None

    # グリッド上のインデックスを計算
    row = (lat - lat_start) / lat_interval
    col = (lon - lon_start) / lon_interval

    row_int, col_int = int(row), int(col)

    # データ範囲外の場合はNoneを返す
    if not (0 <= row_int < geoid_heights.shape[0] - 1 and 0 <= col_int < geoid_heights.shape[1] - 1):
        return None

    # バイリニア補間のための4点を取得
    h11 = geoid_heights[row_int, col_int]
    h12 = geoid_heights[row_int, col_int + 1]
    h21 = geoid_heights[row_int + 1, col_int]
    h22 = geoid_heights[row_int + 1, col_int + 1]

    dx, dy = col - col_int, row - row_int

    # 補間計算
    geoid_h = (h11 * (1 - dx) * (1 - dy) +
               h12 * dx * (1 - dy) +
               h21 * (1 - dx) * dy +
               h22 * dx * dy)

    # 無効値(999.0000)の場合はNoneを返す
    if geoid_h == 999.0000:
        return None

    return geoid_h

def auto_detect_zone(easting, northing):
    """座標値から最も可能性の高い系番号を自動判別する"""
    candidates = []
    # 平面直角座標系の1系から19系までをチェック
    for z_ in range(1, 20):
        epsg_code = 6660 + z_
        try:
            # 座標変換器を作成 (平面直角座標系 -> WGS84)
            transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(easting, northing)
            
            # 変換後の緯度経度が日本の範囲内にあるかチェック
            if (japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and
                japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]):
                candidates.append({"zone": z_, "epsg": epsg_code, "lat": lat, "lon": lon})
        except Exception:
            continue

    if not candidates:
        return None

    # 複数の候補がある場合、日本の中心に近いものを選択
    # ここでは仮に大分県の座標を基準点とする
    reference_point = (33.23, 131.61) 
    for c in candidates:
        c["distance"] = geodesic((c["lat"], c["lon"]), reference_point).meters

    best = min(candidates, key=lambda x: x["distance"])
    best["auto_detected"] = True
    return best

def parse_coordinate_input(input_string):
    """入力文字列から座標リストを抽出する"""
    coordinates = []
    # Define delimiters for separating multiple coordinate sets on a single line
    # Using a regex pattern to split by any of these delimiters
    coordinate_set_delimiters = r',,|,、|  |　　|//' # ,, or 、、 or double half-width space or double full-width space or //

    for line_num, line in enumerate(input_string.splitlines(), 1):
        # Split the line into potential coordinate sets
        # This will handle cases like "X Y Z,,X2 Y2 Z2"
        coordinate_sets_on_line = re.split(coordinate_set_delimiters, line.strip())

        for coord_set_str in coordinate_sets_on_line:
            if not coord_set_str.strip(): # Skip empty strings resulting from split
                continue

            # Extract numbers from each coordinate set string
            found_numbers = re.findall(r'[-+]?\d*\.?\d+', coord_set_str)

            if len(found_numbers) == 3:
                easting = float(found_numbers[0])
                northing = float(found_numbers[1])
                z = float(found_numbers[2])
                coordinates.append({'easting': easting, 'northing': northing, 'z': z})
            elif len(found_numbers) > 3:
                easting = float(found_numbers[0])
                northing = float(found_numbers[1])
                z = float(found_numbers[2])
                coordinates.append({'easting': easting, 'northing': northing, 'z': z})
                st.warning(f"⚠️ {line_num}行目: 座標セット '{coord_set_str}' に3つより多くの数値が見つかりました。最初の3つ ({easting}, {northing}, {z}) を使用します。")
            elif len(found_numbers) > 0:
                st.warning(f"⚠️ {line_num}行目: 座標セット '{coord_set_str}' の数値が3つではありません。スキップされます。")

    return coordinates

# --- Streamlit App ---
st.title("座標変換ツール (平面直角座標系 → 緯度経度)")

# パスワード認証
PASSWORD = os.environ.get("STREAMLIT_PASSWORD", "test") # 環境変数からパスワードを取得、なければ"test"

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
    # ジオイドデータの読み込み
    geoid_heights, lat_start, lon_start, lat_interval, lon_interval = load_geoid_data(GEOID_DATA_PATH)

    if geoid_heights is None:
        st.stop()

    st.markdown("---")
    st.subheader("座標入力")
    st.write("最大500個の座標を一括で変換できます。")

    coordinate_input_text = st.text_area(
        '**X (Easting), Y (Northing), Z (標高)** の順で座標を入力してください.\n\n'
        '1行に1座標ずつ入力するか、または1行に複数の座標を横並びで入力できます。\n'
        '数値はスペース、カンマ、タブなどで区切ってください。\n'
        '複数の座標セットを1行に入力する場合は、`,,`、`、、`、`  `(半角スペース2つ)、`　　`(全角スペース2つ)、`//` のいずれかで区切ってください。\n\n'
        '例:\n'
        '`10000 20000 100`\n'
        '`X=10000, Y=20000, Z=100`\n'
        '`10000 20000 100,, 10000 20000 100`\n'
        '`10000 20000 100  10000 20000 100` (半角スペース2つで区切り)\n'
        '`10000 20000 100// 10000 20000 100`',
        height=250 # Increased height to accommodate more examples
    )

    # 入力オプション
    col1, col2 = st.columns(2)
    with col1:
        zone_input = st.number_input('系番号 (自動判別する場合は 0 を入力):', value=0, min_value=0, max_value=19)
    with col2:
        display_mode = st.radio("表示モードを選択してください:", ("要約表示", "詳細表示"), horizontal=True)

    if st.button('変換実行', type="primary"):
        coordinates_to_convert = parse_coordinate_input(coordinate_input_text)

        if not coordinates_to_convert:
            st.warning("⚠️ 変換対象の座標が入力されていません。")
        elif len(coordinates_to_convert) > 500:
            st.error("⚠️ 一度に変換できる座標は最大500個です。")
        else:
            st.subheader("== 変換結果 (WGS84) ==")
            
            results_data = []
            for i, coord in enumerate(coordinates_to_convert):
                easting, northing, z = coord['easting'], coord['northing'], coord['z']
                
                # 座標値が0の場合はスキップ
                if easting == 0.0 and northing == 0.0:
                    st.warning(f"⚠️ **点 {i+1}**: 座標値が (0, 0) のためスキップしました。")
                    continue

                result_info = None
                # 系番号の決定
                if zone_input == 0: # 自動判別
                    result_info = auto_detect_zone(easting, northing)
                elif 1 <= zone_input <= 19: # 手動指定
                    try:
                        epsg_code = 6660 + zone_input
                        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
                        lon, lat = transformer.transform(easting, northing)
                        
                        # 日本の範囲内かチェック
                        if (japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and
                            japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]):
                            result_info = {"zone": zone_input, "epsg": epsg_code, "lat": lat, "lon": lon, "auto_detected": False}
                        else:
                            # 指定された系で変換すると日本の範囲外になる場合
                            st.warning(f"⚠️ **点 {i+1}**: 指定された第{zone_input}系で変換すると日本の範囲外になります。自動判別を試します。")
                            result_info = auto_detect_zone(easting, northing)

                    except Exception as e:
                        st.error(f"⚠️ **点 {i+1}**: 第{zone_input}系での変換中にエラーが発生しました: {e}")
                        result_info = None

                # 楕円体高の計算
                ellipsoidal_height = None
                geoid_h = None
                if result_info:
                    geoid_h = get_geoid_height(result_info['lat'], result_info['lon'], geoid_heights, lat_start, lon_start, lat_interval, lon_interval)
                    if geoid_h is not None:
                        ellipsoidal_height = z + geoid_h

                # 結果の格納
                results_data.append({
                    "id": i + 1,
                    "input": coord,
                    "result": result_info,
                    "geoid_height": geoid_h,
                    "ellipsoidal_height": ellipsoidal_height
                })

            # 結果の表示
            if display_mode == "詳細表示":
                for res in results_data:
                    st.markdown(f"--- \n### **座標 {res['id']}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**入力座標 (平面直角座標系)**")
                        st.write(f"X (Easting): `{res['input']['easting']}`")
                        st.write(f"Y (Northing): `{res['input']['northing']}`")
                        st.write(f"Z (標高): `{res['input']['z']}`")
                    
                    with col2:
                        st.write("**変換結果 (WGS84)**")
                        if res["result"]:
                            st.write(f"緯度: `{res['result']['lat']:.8f}`")
                            st.write(f"経度: `{res['result']['lon']:.8f}`")
                            if res["ellipsoidal_height"] is not None:
                                st.write(f"ジオイド高: `{res['geoid_height']:.4f} m`")
                                st.write(f"楕円体高: `{res['ellipsoidal_height']:.4f} m`")
                            else:
                                st.warning("ジオイド高取得不可")
                            
                            st.write(f"系番号: `第{res['result']['zone']}系`")
                            if res['result'].get("auto_detected"):
                                st.info("系番号は自動判別されました。")
                        else:
                            st.error("座標変換に失敗しました。入力値が平面直角座標系の範囲内か確認してください。")

            else: # 要約表示
                summary_data = []
                for res in results_data:
                    if res["result"]:
                        h_str = f"{res['ellipsoidal_height']:.3f}" if res['ellipsoidal_height'] is not None else "N/A"
                        zone_str = f"({res['result']['zone']}系)"
                        if res['result'].get("auto_detected"):
                            zone_str = f"({res['result']['zone']}系*)" # 自動判別は*マーク
                        
                        summary_data.append({
                            "点": res["id"],
                            "緯度": f"{res['result']['lat']:.7f}",
                            "経度": f"{res['result']['lon']:.7f}",
                            "楕円体高 (m)": h_str,
                            "系": zone_str
                        })
                    else:
                        summary_data.append({
                            "点": res["id"],
                            "緯度": "変換失敗", "経度": "", "楕円体高 (m)": "", "系": ""
                        })
                
                st.dataframe(summary_data, use_container_width=True)
                st.caption("* が付いている系番号は自動判別されたものです。")