import streamlit as st
from pyproj import Transformer
from geopy.distance import geodesic
import os

# PROJ_DATA環境変数を設定して、pyprojがジオイドモデルファイルを見つけられるようにする
# GSIGEO2011.binがapp.pyと同じディレクトリにあることを想定
os.environ['PROJ_DATA'] = os.path.dirname(__file__)

# 日本の緯度経度範囲
japan_bounds = {
    "lat_min": 20.0,
    "lat_max": 46.0,
    "lon_min": 122.0,
    "lon_max": 154.0
}

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
    # 候補がない場合はNoneを返す
    if not candidates:
        return None
    # 基準点を日本の中心付近に設定し、最も近い候補を選択
    reference_point = (33.5, 131.0)
    for c in candidates:
        c["distance"] = geodesic((c["lat"], c["lon"]), reference_point).meters
    best = min(candidates, key=lambda x: x["distance"])
    best["auto_detected"] = True
    return best

def calculate_geoid_height(lat, lon):
    """
    緯度・経度からジオイド高を計算します。
    GSIGEO2011.binファイルが必要です。
    """
    try:
        # 楕円体高0のWGS84座標を、GSIGEO2011に基づく正標高に変換するパイプライン
        # 出力される正標高は、ジオイド高の符号を反転させたものになる
        transformer = Transformer.from_pipeline(
            "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=vgridshift +grids=GSIGEO2011.bin +step +proj=unitconvert +xy_in=rad +xy_out=deg"
        )
        # 緯度、経度、楕円体高0を入力
        _, _, orthometric_height = transformer.transform(lon, lat, 0)
        # ジオイド高 = 楕円体高 - 正標高
        # ここでは楕円体高を0としているため、ジオイド高 = 0 - orthometric_height = -orthometric_height
        geoid_height = -orthometric_height
        return geoid_height
    except Exception as e:
        st.error(f"ジオイド高の計算中にエラーが発生しました: {e}")
        st.info("GSIGEO2011.binファイルがmy_web_appディレクトリに正しく配置されているか確認してください。")
        return None

st.set_page_config(page_title="座標変換ツール", layout="centered")
st.title("座標変換ツール")

st.write("---")

# 入力ウィジェット
northing = st.number_input('X座標（北ing）:', value=0.0, format="%.3f", help="X座標（北ing）を入力してください。")
easting = st.number_input('Y座標（東ing）:', value=0.0, format="%.3f", help="Y座標（東ing）を入力してください。")
z = st.number_input('Z座標（m）:', value=0.0, format="%.3f", help="Z座標（標高）を入力してください。")
# ジオイド高の入力欄は削除
zone = st.number_input('系番号:', value=0, format="%d", help="1〜19で指定。自動判別は0を入力してください。")

st.write("---")

if st.button('変換実行'):
    if easting == 0 or northing == 0:
        st.warning("⚠️ Y座標（東ing）とX座標（北ing）は0以外を入力してください。")
    else:
        result = None
        if zone == 0:
            result = auto_detect_zone(easting, northing)
        elif 1 <= zone <= 19:
            try:
                epsg_code = 6660 + zone
                transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(easting, northing)
                if (japan_bounds["lat_min"] <= lat <= japan_bounds["lat_max"] and
                    japan_bounds["lon_min"] <= lon <= japan_bounds["lon_max"]):
                    result = {
                        "zone": zone,
                        "epsg": epsg_code,
                        "lat": lat,
                        "lon": lon,
                        "auto_detected": False
                    }
            except Exception:
                result = None

        if result:
            calculated_geoid_height = calculate_geoid_height(result['lat'], result['lon'])
            if calculated_geoid_height is None:
                st.error("ジオイド高の計算に失敗したため、処理を中断します。")
            else:
                ellipsoidal_height = z + calculated_geoid_height
                st.subheader("=== 座標変換結果（WGS84） ===")
                st.write(f"**系番号**: 第{result['zone']}系")
                st.write(f"**EPSGコード**: {result['epsg']}")
                st.write(f"**緯度（北緯）**: {result['lat']}")
                st.write(f"**経度（東経）**: {result['lon']}")
                st.write(f"**計算されたジオイド高**: {calculated_geoid_height}")
                st.write(f"**楕円体高（Z + 計算されたジオイド高）**: {ellipsoidal_height}")
                if result.get("auto_detected", False):
                    st.info("※ 系番号は自動判別されました。")
        else:
            st.error("⚠️ 有効な座標変換結果が得られませんでした。座標または系番号が正しいか確認してください。")