import streamlit as st
from pyproj import Transformer
from geopy.distance import geodesic
import os
import re
import io
import pandas as pd # pandasを再インポート
import numpy as np # numpyをインポート

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
    st.session_state['converted_coords_for_excel'] = None # 追加

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
    """Extracts the first floating-point number from a string, ignoring units and other non-numeric characters."""
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
    セル内の文字列からX,Y,Zのラベルと数値を柔軟に抽出し、座標を認識する。
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
        df_str = df.astype(str)

        # X, Y, Zヘッダーの場所を探す
        x_locs, y_locs, z_locs = [], [], []
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                val = str(df_str.iat[r, c]).lower()
                if not val or val == 'nan': continue
                if re.search(r'x|easting', val): x_locs.append((r, c))
                elif re.search(r'y|northing', val): y_locs.append((r, c))
                elif re.search(r'z|標高|height', val): z_locs.append((r, c))

        used = set()
        header_info = "X, Yのヘッダーが見つかりませんでした。"

        for r_x, c_x in x_locs:
            if (r_x, c_x) in used: continue
            
            # 同じ行にあるYを探す
            y_match = next(((r_y, c_y) for r_y, c_y in y_locs if r_y == r_x and (r_y, c_y) not in used), None)

            if y_match:
                # 同じ行にあるZを探す (オプション)
                z_match = next(((r_z, c_z) for r_z, c_z in z_locs if r_z == r_x and (r_z, c_z) not in used), None)
                
                header_row = r_x
                x_col, y_col = c_x, y_match[1]
                z_col = z_match[1] if z_match else None
                
                block_coords = []
                for r_data in range(header_row + 1, df.shape[0]):
                    try:
                        easting_str = str(df.iat[r_data, x_col]).strip()
                        northing_str = str(df.iat[r_data, y_col]).strip()

                        # X, Yが空欄や数値でない場合はデータの終わりとみなす
                        if not easting_str and not northing_str: break
                        if not easting_str or not northing_str: continue

                        easting = float(easting_str)
                        northing = float(northing_str)

                        z = 0.0
                        if z_col is not None:
                            try:
                                z_str = str(df.iat[r_data, z_col]).strip()
                                if z_str and z_str.lower() != 'nan': # Zが空欄でない場合のみ変換
                                    z = float(z_str)
                            except (ValueError, TypeError):
                                z = 0.0 # Zが数値でない場合は0.0とする

                        block_coords.append({'easting': easting, 'northing': northing, 'z': z})
                    except (ValueError, TypeError, IndexError):
                        # データが数値でない、または範囲外の場合はブロックの終わりと判断
                        break
                
                if block_coords:
                    all_coords.extend(block_coords)
                    used.update({(r_x, c_x), y_match})
                    if z_match:
                        used.update({z_match})
                    header_info = None # 正常に座標が見つかった

        return all_coords, None if all_coords else "X, Yのヘッダーが見つかりませんでした。"
    except Exception as e:
        return None, f"ファイル解析エラー: {e}"

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

# --- ジオイド高結果Excel出力ページ --- 
def geoid_excel_output_page():
    st.header("ジオイド高結果Excel出力")

    # --- Z座標の確認 ---
    z_values = st.session_state.get('z_values_for_geoid')

    if z_values:
        st.success(f"✅ 「座標変換」ページから {len(z_values)}個のZ座標を読み込み済みです。")
    else:
        st.warning("「座標変換」ページで座標を入力・変換すると、ここでZ座標が自動的に読み込まれます。")
        st.info("または、ここで直接Z座標を含むファイルをアップロードすることもできます。")
        
        # フォールバック用のファイルアップローダー
        fallback_file = st.file_uploader("Z座標を含むExcelまたはCSVファイルを選択", type=['xlsx', 'csv', 'xls'], key="z_file_uploader_fallback")
        if fallback_file:
            # フォールバックファイルが使われた場合、そのファイル名を保存
            st.session_state['original_filename'] = fallback_file.name
            coords, z_values_fallback, err = parse_coordinate_file(fallback_file)
            if err:
                st.error(f"⚠️ {err}")
            else:
                st.session_state['z_values_for_geoid'] = z_values_fallback
                st.rerun()
        return # Z値がセットされるまで待機

    # --- .out ファイルのアップロードと処理 ---
    st.subheader("ジオイド高結果 (.out) のアップロードとExcel生成")
    uploaded_out_file = st.file_uploader("国土地理院ジオイド高計算ツールで出力された.outファイルをアップロードしてください", type=["out"], key="out_file_uploader")

    if uploaded_out_file:
        try:
            # .out ファイルの内容を読み込み (UTF-8/Shift-JIS自動判別)
            content_bytes = uploaded_out_file.getvalue()
            try:
                # まずUTF-8でデコードを試みる
                content = content_bytes.decode('utf-8-sig') # BOM付きUTF-8に対応
            except UnicodeDecodeError:
                # UTF-8で失敗した場合、Shift-JISでデコードを試みる
                content = content_bytes.decode('shift-jis')
            
            lines = content.splitlines()

            # ヘッダー行をスキップし、データ行を解析
            data_rows = []
            for line in lines:
                if not line.strip() or line.strip().startswith('#') or line.strip().startswith('-'):
                    continue # コメント行や区切り線をスキップ
                
                parts = line.strip().split() # 半角空白で分割
                if len(parts) >= 5: # 緯度, 経度, ジオイド高, 基準面補正量, ジオイド高+基準面補正量
                    try:
                        lat_dms = parts[0]
                        lon_dms = parts[1]
                        geoid_height_plus_correction = float(parts[4]) # ジオイド高+基準面補正量
                        
                        # DMSを10進数に変換
                        lat_decimal = dms_string_to_decimal(lat_dms)
                        lon_decimal = dms_string_to_decimal(lon_dms)

                        if lat_decimal is not None and lon_decimal is not None:
                            data_rows.append({
                                "緯度(dms)": lat_dms,
                                "経度(dms)": lon_dms,
                                "緯度(10進)": lat_decimal,
                                "経度(10進)": lon_decimal,
                                "ジオイド高+基準面補正量(m)": geoid_height_plus_correction
                            })
                        else:
                            st.warning(f"⚠️ DMSから10進数への変換に失敗しました: {line}")

                    except (ValueError, IndexError):
                        st.warning(f"⚠️ データ行の解析に失敗しました: {line}")
                        continue
            
            if not data_rows:
                st.warning("⚠️ .out ファイルから有効なデータが見つかりませんでした。ファイル形式を確認してください。")
                return

            # 座標変換ページで保存された元のZ座標を取得
            original_z_coords = st.session_state.get('z_values_for_geoid', [])

            output_data = []
            for i, row_data in enumerate(data_rows):
                lat_dms = row_data["緯度(dms)"]
                lon_dms = row_data["経度(dms)"]
                geoid_plus_correction = row_data["ジオイド高+基準面補正量(m)"]

                # 元のZ座標を取得 (順序が一致している前提)
                original_z = 0.0
                if i < len(original_z_coords):
                    original_z = original_z_coords[i]
                
                ellipsoidal_height = original_z + geoid_plus_correction

                output_data.append({
                    "緯度(dms)": lat_dms,
                    "経度(dms)": lon_dms,
                    "緯度(10進)": f"{row_data['緯度(10進)']:.10f}",
                    "経度(10進)": f"{row_data['経度(10進)']:.10f}",
                    "楕円体高(m)": f"{ellipsoidal_height:.4f}"
                })

            output_df = pd.DataFrame(output_data)
            st.dataframe(output_df, use_container_width=True)

            # Excelダウンロードボタン
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                output_df.to_excel(writer, index=False, sheet_name='ジオイド高結果')
                
                # ここから列幅自動調整処理
                worksheet = writer.sheets['ジオイド高結果']
                for column_cells in worksheet.columns:
                    # 各列の最大文字数を計算 (ヘッダーも含む)
                    max_length = 0
                    column_letter = column_cells[0].column_letter # 列のアルファベットを取得
                    for cell in column_cells:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    # 少し余裕を持たせた幅に調整
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            processed_data = output.getvalue()

            # 動的なファイル名を生成
            original_filename = st.session_state.get('original_filename', 'default_results.xlsx')
            output_filename = f"変換_{original_filename}"

            st.download_button(
                label="Excelファイル (.xlsx) をダウンロード",
                data=processed_data,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                on_click=clear_download_state # ダウンロードボタンが押されたらコールバックを実行
            )

        except Exception as e:
            st.error(f"⚠️ ファイルの処理中にエラーが発生しました: {e}")
            st.info("ファイル形式が正しいか、またはShift-JISエンコードであることを確認してください。")

# --- Streamlit App --- 
PASSWORD = "test" # パスワードをハードコード

def main():
    if 'password_correct' not in st.session_state:
        st.session_state['password_correct'] = False

    if not st.session_state['password_correct']:
        st.title("ログイン")
        password_input = st.text_input("パスワードを入力してください", type="password")
        if st.button("ログイン"):
            if password_input == PASSWORD:
                st.session_state['password_correct'] = True
                st.rerun()
            else:
                st.error("パスワードが間違っています")
    else:
        st.sidebar.title("ナビゲーション")
        page = st.sidebar.radio("ページを選択", ["座標変換", "変換結果出力"])

        if page == "座標変換":
            st.title("座標変換ツール (JGD2011平面直角座標系 → WGS84緯度経度)")

            st.header("入力方法の選択")
            input_method = st.radio("座標の入力方法を選択してください:", ("ファイルアップロード", "テキスト入力"), horizontal=True)

            coordinates_to_convert = []
            uploaded_file = None
            coordinate_input_text = ""

            if input_method == "ファイルアップロード":
                uploaded_file = st.file_uploader("座標ファイル (CSV/Excel) をアップロードしてください", type=["csv", "xlsx", "xls"])
                if uploaded_file:
                    st.info("ファイルからX, Y, Z座標を自動的に検出します。")
            else:
                coordinate_input_text = st.text_area(
                    "座標をテキストで入力してください (X Y Z または X Y の形式で複数行可、スペース区切り):",
                    ""
                )

            st.header("変換設定")
            selected_zone = st.selectbox(
                "平面直角座標系の系を選択してください:",
                ["自動検出"] + [f"{i}系" for i in range(1, 20)]
            )

            display_mode = st.radio("表示モード:", ("要約表示", "詳細表示"), horizontal=True)

            if st.button('変換実行', type="primary"):
                # 変換実行時にファイル名をリセット
                st.session_state['original_filename'] = None
                st.session_state['converted_coords_for_excel'] = [] # Excel出力用に変換結果を保存するリスト

                if input_method == "ファイルアップロード":
                    if uploaded_file:
                        # 元のファイル名をセッションステートに保存
                        st.session_state['original_filename'] = uploaded_file.name
                        with st.spinner('ファイルを処理中...'):
                            coords, err = parse_coordinate_file(uploaded_file)
                            if err:
                                st.error(f"⚠️ {err}")
                                st.session_state['converted_coords'] = []
                            elif coords:
                                st.success(f"✅ {len(coords)}個の座標を読み込みました。")
                                st.session_state['converted_coords'] = []
                                st.session_state['z_values_for_geoid'] = [] # ジオイド高計算用にZ値を保存

                                for i, coord in enumerate(coords):
                                    easting = coord['easting']
                                    northing = coord['northing']
                                    z_value = coord.get('z', 0.0) # Z値がない場合は0.0

                                    zone_info = None
                                    if selected_zone == "自動検出":
                                        zone_info = auto_detect_zone(easting, northing)
                                        if not zone_info:
                                            st.warning(f"⚠️ 座標 ({easting}, {northing}) の系を自動検出できませんでした。スキップします。")
                                            continue
                                    else:
                                        zone_num = int(selected_zone.replace("系", ""))
                                        zone_info = {"zone": zone_num, "epsg": 6660 + zone_num, "auto_detected": False}

                                    try:
                                        transformer = Transformer.from_crs(f"EPSG:{zone_info['epsg']}", "EPSG:4326", always_xy=True)
                                        lon, lat = transformer.transform(easting, northing)

                                        # 楕円体高の計算 (Z座標 + ジオイド高) - ここではジオイド高はまだ不明なので、Z座標をそのまま渡す
                                        # ジオイド高はgeoid_excel_output_pageで処理されるため、ここではZ座標をそのまま保存
                                        ellipsoidal_height_display = f"Z={z_value:.3f}m" # 初期表示はZ座標

                                        result = {
                                            "input": {"easting": easting, "northing": northing, "z": z_value},
                                            "output": {"lat": lat, "lon": lon, "ellipsoidal_height": ellipsoidal_height_display},
                                            "zone_info": zone_info
                                        }
                                        st.session_state['converted_coords'].append(result)
                                        st.session_state['z_values_for_geoid'].append(z_value) # Z値をリストに追加
                                        st.session_state['converted_coords_for_excel'].append(result) # Excel出力用に結果を保存

                                    except Exception as e:
                                        st.error(f"変換エラー (X: {easting}, Y: {northing}, 系: {zone_info['zone']}系): {e}")
                                        continue
                            else:
                                st.warning("⚠️ ファイルから有効な座標データを抽出できませんでした。")
                    else:
                        st.warning("⚠️ ファイルが選択されていません。")
                else: # テキスト入力の場合
                    coordinates_to_convert = parse_coordinate_text(coordinate_input_text)
                    if coordinates_to_convert:
                        st.session_state['converted_coords'] = []
                        st.session_state['z_values_for_geoid'] = [] # ジオイド高計算用にZ値を保存
                        st.session_state['converted_coords_for_excel'] = [] # Excel出力用に変換結果を保存するリスト
                        # テキスト入力用のデフォルトファイル名を設定
                        st.session_state['original_filename'] = "text_input.xlsx"

                        for i, coord in enumerate(coordinates_to_convert):
                            easting = coord['easting']
                            northing = coord['northing']
                            z_value = coord.get('z', 0.0) # Z値がない場合は0.0

                            zone_info = None
                            if selected_zone == "自動検出":
                                zone_info = auto_detect_zone(easting, northing)
                                if not zone_info:
                                    st.warning(f"⚠️ 座標 ({easting}, {northing}) の系を自動検出できませんでした。スキップします。")
                                    continue
                            else:
                                zone_num = int(selected_zone.replace("系", ""))
                                zone_info = {"zone": zone_num, "epsg": 6660 + zone_num, "auto_detected": False}

                            try:
                                transformer = Transformer.from_crs(f"EPSG:{zone_info['epsg']}", "EPSG:4326", always_xy=True)
                                lon, lat = transformer.transform(easting, northing)

                                ellipsoidal_height_display = f"Z={z_value:.3f}m" # 初期表示はZ座標

                                result = {
                                    "input": {"easting": easting, "northing": northing, "z": z_value},
                                    "output": {"lat": lat, "lon": lon, "ellipsoidal_height": ellipsoidal_height_display},
                                    "zone_info": zone_info
                                }
                                st.session_state['converted_coords'].append(result)
                                st.session_state['z_values_for_geoid'].append(z_value) # Z値をリストに追加
                                st.session_state['converted_coords_for_excel'].append(result) # Excel出力用に結果を保存

                            except Exception as e:
                                st.error(f"変換エラー (X: {easting}, Y: {northing}, 系: {zone_info['zone']}系): {e}")
                                continue
                    else:
                        st.warning("⚠️ 変換対象の座標が見つかりませんでした。")

            if 'converted_coords' in st.session_state and st.session_state['converted_coords']:
                st.subheader("変換結果")
                if display_mode == "要約表示":
                    summary_data = []
                    for coord in st.session_state['converted_coords']:
                        summary_data.append({
                            "入力X": f"{coord['input']['easting']:.3f}",
                            "入力Y": f"{coord['input']['northing']:.3f}",
                            "入力Z": f"{coord['input']['z']:.3f}",
                            "出力緯度": f"{coord['output']['lat']:.10f}",
                            "出力経度": f"{coord['output']['lon']:.10f}",
                            "楕円体高": coord['output']['ellipsoidal_height'], # Z値そのまま
                            "系": f"{coord['zone_info']['zone']}系" + (" (自動検出)" if coord['zone_info']['auto_detected'] else "")
                        })
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                else: # 詳細表示
                    for i, coord in enumerate(st.session_state['converted_coords']):
                        st.write(f"--- 座標 {i+1} ---")
                        st.write(f"**入力座標:**")
                        st.write(f"  X: {coord['input']['easting']:.3f}")
                        st.write(f"  Y: {coord['input']['northing']:.3f}")
                        st.write(f"  Z: {coord['input']['z']:.3f}")
                        st.write(f"**検出された系:** {coord['zone_info']['zone']}系" + (" (自動検出)" if coord['zone_info']['auto_detected'] else ""))
                        st.write(f"**変換結果 (WGS84):**")
                        st.write(f"  緯度: {coord['output']['lat']:.10f}")
                        st.write(f"  経度: {coord['output']['lon']:.10f}")
                        st.write(f"  楕円体高: {coord['output']['ellipsoidal_height']}") # Z値そのまま

        elif page == "ジオイド高結果Excel出力":
            geoid_excel_output_page()

if __name__ == "__main__":
    main()
