'''
    mapping geo coordinates to s2geometry cell
'''
import s2geometry as s2

def geo2cell(lat, lon, level):
    try:
        p = s2.S2LatLng.FromDegrees(lat, lon)
        leaf = s2.S2CellId(p)
        cell = leaf.parent(level)
        return cell
    except Exception as ex:
        print(f'geo2cell exception {ex}')
        return None

if __name__ == '__main__':
    lat = 51.3368602
    lon = 0.4931979
    print(f'lat:{lat} lon:{lon}')
    for level in range(31):
        cell = geo2cell(lat=lat, lon=lon, level=level)
        print(f'level:{level} cell-id:{cell.id()} token:{cell.ToToken()} str:{cell.ToString()} '
              f'area:{s2.S2Cell(cell).ExactArea()}')