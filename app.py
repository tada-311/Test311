@ -1,246 +1,247 @@
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

                    st.warning("⚠️ 入力された数値から有効な座標ペアを抽出できませんでした。X (Northing), Y (Easting), (Z) の形式で入力されているか確認してください。")

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
        st.warning("⚠️ 入力された数値から有効な座標ペアを抽出できませんでした。X (Northing), Y (Easting), (Z) の形式で入力されているか@ -1,116 +1,129 @@
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
@ -178,17 +191,17 @@ def parse_coordinate_file(uploaded_file):
                    if not easting_str and not northing_str: break
                    if not easting_str or not northing_str: continue

                    easting = float(easting_str)
                    northing = float(northing_str)
                    easting = _extract_float_from_string(easting_str)
                    northing = _extract_float_from_string(northing_str)

                    if easting is None or northing is None: continue

                    z = 0.0
                    if z_col is not None:
                        try:
                            z_str = str(df.iat[r_data, z_col]).strip()
                            if z_str and z_str.lower() != 'nan':
                                z = float(z_str)
                        except (ValueError, TypeError):
                            z = 0.0
                        z_str = str(df.iat[r_data, z_col]).strip()
                        z_val = _extract_float_from_string(z_str)
                        if z_val is not None:
                            z = z_val

                    all_coords.append({'easting': easting, 'northing': northing, 'z': z})
                    all_z_values.append(z)
確認してください。")

    return coordinates

# --- Z座標抽出関数 ---



# --- ジオイド高結果Excel出力ページ --- 
def geoid_excel_output_page():
    st.header("ジオイド高結果Excel出力")

    # --- Z座標の確認 ---
    z_values = st.session_state.get('z_values_for_geoid')

    if z_values:
        st.success(f"✅ 「座標変換」ページから {len(z_values)}個のZ座標を読み込み済みです。")
        filename_display = st.session_state.get('original_filename', '入力データ') # ファイル名がなければ「入力データ」
        st.success(f"✅ 「{filename_display}」から {len(z_values)}個のZ座標を読み込み済みです。")
    else:
        st.warning("「座標変換」ページで座標を入力・変換すると、ここでZ座標が自動的に読み込まれます。")
        st.info("または、ここで直接Z座標を含むファイルをアップロードすることもできます。")
