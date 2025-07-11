import streamlit as st
from pyproj import Transformer
from geopy.distance import geodesic
import os
import re
import io
import pandas as pd # pandasを再インポート

# --- バージョン情報: 2024-07-10_v3.0 - 座標順序修正とジオイド高Excel出力機能追加 ---

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

def dms_string_to_decimal(dms_string):
    """度分秒形式の文字列（例: 333437.14801）を10進数の度に変換"""
    try:
        dms_string = str(dms_string).strip()
        if '.' not in dms_string:
            dms_string += '.0'
        
        parts = dms_string.split('.')
        integer_part = parts[0]
        decimal_part = parts[1]

        # 整数部が6文字未満（例: 333437 -> DDMMSS）の場合はエラーとするか、適切に処理する必要がある
        if len(integer_part) < 6:
            return None

        # 秒 (SS.sssss)
        seconds_str = integer_part[-2:] + '.' + decimal_part
        seconds = float(seconds_str)
        
        # 分 (MM)
        minutes_str = integer_part[-4:-2]
        minutes = int(minutes_str)
        
        # 度 (DD or DDD)
        degrees_str = integer_part[:-4]
        degrees = int(degrees_str)
        
        decimal_degrees = degrees + minutes / 60 + seconds / 3600
        return decimal_degrees
    except (ValueError, IndexError):
        return None

# --- セッション状態クリア関数 ---
def clear_download_state():
    """ダウンロード後にセッション情報をクリアするコールバック関数"""
    st.session_state['z_values_for_geoid'] = None
    st.session_state['original_filename'] = None
    st.session_state['conversion_results_for_geoid'] = None
    st.session_state['uploaded_file_data'] = None
    st.session_state['uploaded_file_content'] = None

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
def _extract_float_from_string(s):
    """Extracts the first floating-point number from a string, ignoring units."""
    if not isinstance(s, str):
        return None
    # This regex finds the first float-like number in the string
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        try:
            return float(match.group(0))
        except (ValueError, TypeError):
            return None
    return None

def parse_coordinate_file(uploaded_file):
    """
    アップロードされたファイル（Excel/CSV）を解析し、座標データとZ値のリストを返す統一関数。
    X,Yのペアを基準にデータを読み込み、複数のブロックに対応する。
    戻り値: (all_coords, all_z_values, error_message)
    """
    if uploaded_file is None:
        return None, None, "ファイルがアップロードされていません。"
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
            return None, None, "サポートされていないファイル形式です。"

        all_coords = []
        all_z_values = []
        df_str = df.astype(str)

        # Find all X, Y, Z header locations
        x_locs, y_locs, z_locs = [], [], []
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                val = str(df_str.iat[r, c]).lower()
                if not val or val == 'nan': continue
                if re.search(r'x|easting', val): x_locs.append((r, c))
                elif re.search(r'y|northing', val): y_locs.append((r, c))
                elif re.search(r'z|標高|height', val): z_locs.append((r, c))

        if not x_locs or not y_locs:
            return None, None, "XおよびYのヘッダーが見つかりませんでした。"

        used_headers = set()
        x_locs.sort() # Process headers in a predictable order

        for r_x, c_x in x_locs:
            if (r_x, c_x) in used_headers: continue
            
            y_candidates = [loc for loc in y_locs if loc[0] == r_x and loc not in used_headers]
            if not y_candidates: continue
            y_match = y_candidates[0]

            z_candidates = [loc for loc in z_locs if loc[0] == r_x and loc not in used_headers]
            z_match = z_candidates[0] if z_candidates else None
            
            header_row = r_x
            x_col, y_col = c_x, y_match[1]
            z_col = z_match[1] if z_match else None
            
            used_headers.add((r_x, c_x))
            used_headers.add(y_match)
            if z_match: used_headers.add(z_match)

            for r_data in range(header_row + 1, df.shape[0]):
                try:
                    easting_str = str(df.iat[r_data, x_col]).strip()
                    northing_str = str(df.iat[r_data, y_col]).strip()

                    if not easting_str and not northing_str: break
                    if not easting_str or not northing_str: continue

                    easting = _extract_float_from_string(easting_str)
                    northing = _extract_float_from_string(northing_str)

                    if easting is None or northing is None: continue

                    z = 0.0
                    if z_col is not None:
                        z_str = str(df.iat[r_data, z_col]).strip()
                        z_val = _extract_float_from_string(z_str)
                        if z_val is not None:
                            z = z_val

                    all_coords.append({'easting': easting, 'northing': northing, 'z': z})
                    all_z_values.append(z)

                except (ValueError, TypeError, IndexError):
                    break
        
        if not all_coords:
            return None, None, "有効な座標データが見つかりませんでした。"

        return all_coords, all_z_values, None
    except Exception as e:
        return None, None, f"ファイル解析エラー: {e}"

def parse_coordinate_text(input_string):
    coordinates = []
    # 入力文字列全体からすべての数値を抽出
    all_numbers_str = re.findall(r'[-+]?\d*\.?\d+', input_string)
    all_numbers = [float(n) for n in all_numbers_str]

    i = 0
    while i < len(all_numbers):
        if i + 1 < len(all_numbers): # X (Northing), Y (Easting) が少なくともある場合
            northing_val = all_numbers[i]  # ユーザーのXはNorthing
            easting_val = all_numbers[i+1] # ユーザーのYはEasting
            z = 0.0 # デフォルトZ値

            if i + 2 < len(all_numbers): # Z もある場合
                z = all_numbers[i+2]
                coordinates.append({'easting': easting_val, 'northing': northing_val, 'z': z})
                i += 3 # 3つ進む
            else:
                # Zがない場合 (X, Y のみ)
                coordinates.append({'easting': easting_val, 'northing': northing_val, 'z': z})
                i += 2 # 2つ進む
        else:
            # 数値がXのみでYがない場合など、不完全な座標
            st.warning(f"⚠️ 不完全な座標データが見つかりました。スキップされます。残りの数値: {all_numbers[i:]}")
            break # 残りの数値は処理できないのでループを抜ける

    if not coordinates and all_numbers:
        st.warning("⚠️ 入力された数値から有効な座標ペアを抽出できませんでした。X (Northing), Y (Easting), (Z) の形式で入力されているか確認してください。")

    return coordinates

# --- Z座標抽出関数 ---



# --- ジオイド高結果Excel出力ページ --- 
def geoid_excel_output_page():
    st.header("ジオイド高結果Excel出力")

    # --- Z座標の確認 ---
    z_values = st.session_state.get('z_values_for_geoid')
    conversion_results = st.session_state.get('conversion_results_for_geoid')
    filename_display = st.session_state.get('original_filename', '入力データ') # ファイル名がなければ「入力データ」

    if z_values:
        st.success(f"✅ 「座標変換」ページから {len(z_values)}個のZ座標を読み込み済みです。")
        filename_display = st.session_state.get('original_filename', '入力データ') # ファイル名がなければ「入力データ」
        st.success(f"✅ 「{filename_display}」から {len(z_values)}個のZ座標を読み込み済みです。")
    else:
        st.warning("「座標変換」ページで座標を入力・変換すると、ここでZ座標が自動的に読み込まれます。")
        st.info("または、ここで直接Z座標を含むファイルをアップロードすることもできます。")

    uploaded_geoid_file = st.file_uploader("ジオイド高計算結果ファイル (.out) をアップロード", type=['out'])

    if uploaded_geoid_file and conversion_results:
        try:
            try:
                geoid_content = uploaded_geoid_file.getvalue().decode('utf-8')
            except UnicodeDecodeError:
                geoid_content = uploaded_geoid_file.getvalue().decode('shift-jis')
            geoid_heights = []
            for line in geoid_content.splitlines():
                line = line.strip()
                if not line or line.startswith('#'): # 空行またはコメント行をスキップ
                    continue
                parts = line.split()
                if len(parts) >= 5: # 少なくとも5つの要素があることを確認
                    try:
                        geoid_heights.append(float(parts[4])) # 5番目の要素（ジオイド高+基準面補正量）を抽出
                    except ValueError:
                        continue # 数値に変換できない場合はスキップ
            
            if len(geoid_heights) != len(conversion_results):
                st.error(f"⚠️ ジオイド高の数 ({len(geoid_heights)}) が変換された座標の数 ({len(conversion_results)}) と一致しません。")
            else:
                output_data = []
                for i, res in enumerate(conversion_results):
                    original_z = res["original_z"]
                    geoid_h = geoid_heights[i]
                    ellipsoid_h = original_z + geoid_h
                    output_data.append({
                        "緯度": f"{res["result"]["lat"]}",
                        "経度": f"{res["result"]["lon"]}",
                        "楕円体高": ellipsoid_h
                    })
                
                output_df = pd.DataFrame(output_data)
                st.dataframe(output_df, use_container_width=True)

                # Excelダウンロード
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    output_df.to_excel(writer, index=False, sheet_name='ジオイド高結果')
                output.seek(0)
                st.download_button(
                    label="結果をExcelでダウンロード",
                    data=output.getvalue(),
                    file_name=f"変換_{filename_display.replace('.xlsx', '').replace('.xls', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    on_click=clear_download_state # ダウンロード後にセッションをクリア
                )
        except Exception as e:
            st.error(f"ジオイド高ファイルの処理中にエラーが発生しました: {e}")
    elif uploaded_geoid_file and not conversion_results:
        st.warning("ジオイド高計算結果ファイルを処理するには、まず「座標変換」ページで座標を変換し、Z座標を読み込む必要があります。")

# --- メインアプリのロジック ---
def main_app():
    st.subheader("座標入力")
    input_method = st.radio("入力方法を選択:", ("ファイルアップロード", "テキスト入力"), horizontal=True)
    
    coordinate_input_text = ""
    uploaded_file = None

    if input_method == "ファイルアップロード":
        st.info("Excel (.xlsx) または CSV (.csv) ファイルをアップロードしてください。")
        
        # ファイルアップローダーを表示
        uploaded_file_obj = st.file_uploader(
            "ファイルを選択", 
            type=['xlsx', 'csv'], 
            key="file_uploader_key"
        )
        
        # 新しいファイルがアップロードされた場合、またはファイルがクリアされた場合にセッション状態を更新
        if uploaded_file_obj is not None:
            st.session_state['uploaded_file_content'] = uploaded_file_obj.getvalue()
            st.session_state['uploaded_file_name'] = uploaded_file_obj.name
        elif 'uploaded_file_content' not in st.session_state:
            st.session_state['uploaded_file_content'] = None
            st.session_state['uploaded_file_name'] = None

        # 変換処理で使用するファイルデータを設定
        if st.session_state['uploaded_file_content'] is not None:
            uploaded_file = io.BytesIO(st.session_state['uploaded_file_content'])
            uploaded_file.name = st.session_state['uploaded_file_name']
        else:
            uploaded_file = None
    else:
        coordinate_input_text = st.text_area(
            '**X, Y** の順で座標を入力してください。\n\n'
            '1行に1座標ずつ入力します。数値はスペース、カンマ、タブなどで区切ってください。\n\n'
            '例:\n'
            '`-36258.580  -147524.100`\n'
            '`X=-36258.580, Y=-147524.100`',
            height=150,
            key="coordinate_input_text_area" # セッション状態に紐付けるキーを追加
        )

    col1, col2 = st.columns(2)
    with col1:
        zone_input = st.number_input('系番号 (自動判別は 0):', value=0, min_value=0, max_value=19)
    with col2:
        display_mode = st.radio("表示モード:", ("要約表示", "詳細表示"), horizontal=True)

    if st.button('変換実行', type="primary"):
        coordinates_to_convert = []
        z_values_from_conversion = []

        if input_method == "ファイルアップロード":
            if uploaded_file:
                with st.spinner('ファイルを処理中...'):
                    coords, z_vals, err = parse_coordinate_file(uploaded_file)
                if err:
                    st.error(f"⚠️ {err}")
                else:
                    coordinates_to_convert = coords
                    z_values_from_conversion = z_vals
                    st.session_state['original_filename'] = uploaded_file.name # ファイル名をセッションに保存
            else:
                st.warning("⚠️ ファイルが選択されていません。")
        else:
            coordinates_to_convert = parse_coordinate_text(coordinate_input_text)
            # テキスト入力の場合、Z値は0として扱うか、別途入力させる必要があるが、ここでは簡易的に0とする
            z_values_from_conversion = [coord.get('z', 0.0) for coord in coordinates_to_convert]

        if not coordinates_to_convert:
            st.warning("⚠️ 変換対象の座標が見つかりませんでした。")
        elif len(coordinates_to_convert) > 500:
            st.error(f"⚠️ 最大500個まで。(現在 {len(coordinates_to_convert)}個)")
        else:
            st.subheader("== 変換結果 (WGS84) ==")
            results_data = []
            progress_bar = st.progress(0)
            
            for i, coord in enumerate(coordinates_to_convert):
                easting, northing, original_z = coord['easting'], coord['northing'], coord['z']
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

                results_data.append({"id": i + 1, "input": coord, "result": result_info, "original_z": original_z})
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
                        st.write(f"Z座標（標高）: `{res['original_z']:.3f}`") # 元のZ座標を表示
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
                            "系": zone_str,
                            "Z座標（標高）": f"{res['original_z']:.3f}" # 要約表示にも元のZ座標を追加
                        })
                    else:
                        summary_data.append({"点": res["id"], "緯度": "変換失敗", "経度": "", "系": "", "元のZ": ""})
                st.dataframe(summary_data, use_container_width=True)
                st.caption("\* が付いている系番号は自動判別されたものです。")

            # ジオイド高計算用ファイル出力ボタン
            geoid_in_content = "# 緯度(dms)   経度(dms)\n"
            for res in results_data:
                if res["result"]:
                    lat_dms = decimal_to_dms_string(res["result"]["lat"])
                    lon_dms = decimal_to_dms_string(res["result"]["lon"])
                    geoid_in_content += f"{lat_dms} {lon_dms}\n"
            
            if geoid_in_content != "# 緯度(dms)   経度(dms)
": # ヘッダー行以外にデータがある場合のみボタンを表示
                st.download_button(
                    label="ジオイド高計算用ファイル (.in) をダウンロード",
                    data=geoid_in_content.encode('shift-jis'), # Shift-JISでエンコード
                    file_name="geoid.in",
                    mime="text/plain"
                )
            
            # Z座標と変換結果をセッションに保存
            st.session_state['z_values_for_geoid'] = z_values_from_conversion
            st.session_state['conversion_results_for_geoid'] = results_data
page_selection = st.sidebar.radio("ページ選択", ["X,Y座標の変換", "楕円体高計算と座標のExcel出力"])

if page_selection == "X,Y座標の変換":
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
        main_app()
elif page_selection == "楕円体高計算と座標のExcel出力":
    geoid_excel_output_page()