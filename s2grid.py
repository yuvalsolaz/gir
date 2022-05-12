
## cmake -DWITH_GFLAGS=ON -DOPENSSL_INCLUDE_DIR=openssl_include_dir ..

'''
    transform coordinate to s2 category

'''
"""
This example shows how to add spatial data to an information retrieval
system.  Such systems work by converting documents into a collection of
"index terms" (e.g., representing words or phrases), and then building an
"inverted index" that maps each term to a list of documents (and document
positions) where that term occurs.

This example shows how to convert spatial data into index terms, which can
then be indexed along with other document information.
This is a port of the C++ term_index.cc example for the Python API.
"""

'''
    Usage Example:
    london = s2.S2LatLngRect(s2.S2LatLng.FromDegrees(51.3368602, 0.4931979),
                         s2.S2LatLng.FromDegrees(51.7323965, 0.1495211))
    e14lj = s2.S2LatLngRect(s2.S2LatLng.FromDegrees(51.5213527, -0.0476026),
                    s2.S2LatLng.FromDegrees(51.5213527, -0.0476026))
    print(f'london contains e14lj: {london.Contains(e14lj)}')
     
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

import sys
if __name__ == '__main__':
    level = 4
    lat = 51.3368602
    lon = 0.4931979
    if len(sys.argv) > 1:
        level = int(sys.argv[1])
    cell = geo2cell(lat=lat, lon=lon, level=level)
    print(f'lat:{lat} lon:{lon} level:{level} area:{s2.S2Cell(cell).ExactArea()}')
    print(f'label:{cell.id()} {cell.ToToken()} {cell.ToString()}')