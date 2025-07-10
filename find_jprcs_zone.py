from pyproj import Transformer
from math import isclose

expected_lat = 33.4608197824
expected_lon = 131.1157985691
input_easting = 51109.748
input_northing = 10764.069

tolerance = 0.01 # 許容誤差 (メートル単位)

print(f"Searching for JPRCS zone for input Easting: {input_easting}, Northing: {input_northing}")
print(f"Expected WGS84 Lat: {expected_lat}, Lon: {expected_lon}")

found_zone = None
for zone in range(1, 20):
    epsg_code = 6660 + zone
    try:
        # WGS84からJPRCSへの逆変換
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
        transformed_easting, transformed_northing = transformer.transform(expected_lon, expected_lat)

        # 変換結果と入力値を比較
        if isclose(transformed_easting, input_easting, abs_tol=tolerance) and \
           isclose(transformed_northing, input_northing, abs_tol=tolerance):
            found_zone = zone
            print(f"Match found! Zone: {zone}")
            print(f"  Transformed Easting: {transformed_easting:.3f}, Northing: {transformed_northing:.3f}")
            break
    except Exception as e:
        # print(f"Zone {zone}: Error during transformation - {e}")
        pass

if found_zone:
    print(f"The input coordinates likely belong to JPRCS Zone {found_zone}.")
else:
    print("No matching JPRCS zone found for the given input and expected output.")
