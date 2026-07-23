"""Geographic metadata and edge helpers for the 20-clinic AuCT network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Any, Sequence

from src.graph.edges import Edge, complete_undirected_edges

Coordinate = tuple[float, float]


@dataclass(frozen=True)
class ClinicLocation:
    """Location metadata for one point-of-care treatment center."""

    index: int
    name: str
    city: str
    state: str
    address: str
    latitude: float
    longitude: float
    distance_to_ca_miles: float
    duration_to_ca_hours: float
    cost_to_ca_usd: float
    supplier: int


BB_MANUFACTURING_SITE = {
    "name": "Kite Pharma",
    "city": "El Segundo",
    "state": "California",
    "address": "2355 Utah Ave, El Segundo, CA 90245 USA",
    "latitude": 33.9164,
    "longitude": -118.3990,
}


BB_SUPPLIERS = (
    {
        "id": 1,
        "city": "San Jose",
        "state": "California",
        "latitude": 37.3382,
        "longitude": -121.8863,
    },
    {
        "id": 2,
        "city": "Waltham",
        "state": "Massachusetts",
        "latitude": 42.3765,
        "longitude": -71.2356,
    },
)


BB_20_CLINIC_LOCATIONS: tuple[ClinicLocation, ...] = (
    ClinicLocation(
        1,
        "Seattle Cancer Care Alliance",
        "Seattle",
        "Washington",
        "825 Eastlake Ave E, Seattle, WA 98109",
        47.6263,
        -122.3308,
        1200.0,
        2.8333,
        312.0,
        1,
    ),
    ClinicLocation(
        2,
        "Oregon Health and Science University Hospital",
        "Portland",
        "Oregon",
        "3181 SW Sam Jackson Park Rd, Portland, OR 97239",
        45.4993,
        -122.6867,
        1000.0,
        2.3333,
        287.0,
        1,
    ),
    ClinicLocation(
        3,
        "Stanford Health Care",
        "Palo Alto",
        "California",
        "211 Quarry Rd #302, Palo Alto, CA 94304",
        37.4336,
        -122.1750,
        400.0,
        1.25,
        97.0,
        1,
    ),
    ClinicLocation(
        4,
        "UCLA Health",
        "Los Angeles",
        "California",
        "700 W 7th St S270-D, Los Angeles, CA 90017",
        34.0485,
        -118.2577,
        20.0,
        0.5833,
        0.0,
        1,
    ),
    ClinicLocation(
        5,
        "UC San Diego Health",
        "San Diego",
        "California",
        "200 W Arbor Dr, San Diego, CA 92103",
        32.7530,
        -117.1650,
        120.0,
        2.1333,
        0.0,
        1,
    ),
    ClinicLocation(
        6,
        "Mayo Clinic Arizona",
        "Phoenix",
        "Arizona",
        "5701 E Mayo Blvd, Phoenix, AZ 85054",
        33.6584,
        -111.9565,
        485.0,
        1.25,
        281.0,
        1,
    ),
    ClinicLocation(
        7,
        "Colorado Blood Cancer Institute",
        "Denver",
        "Colorado",
        "1721 E 19th Ave #300, Denver, CO 80218",
        39.7477,
        -104.9670,
        1100.0,
        2.25,
        252.0,
        1,
    ),
    ClinicLocation(
        8,
        "Baylor University Medical Center / Texas Oncology",
        "Dallas",
        "Texas",
        "3410 Worth St Suite 400, Dallas, TX 75246",
        32.7899,
        -96.7794,
        1500.0,
        2.8333,
        285.0,
        1,
    ),
    ClinicLocation(
        9,
        "Mayo Clinic",
        "Rochester",
        "Minnesota",
        "200 1st St SW, Rochester, MN 55905",
        44.0227,
        -92.4669,
        2000.0,
        5.5,
        374.0,
        2,
    ),
    ClinicLocation(
        10,
        "Siteman Cancer Center",
        "St. Louis",
        "Missouri",
        "4921 Parkview Pl, St. Louis, MO 63110",
        38.6368,
        -90.2630,
        1900.0,
        3.4167,
        464.0,
        2,
    ),
    ClinicLocation(
        11,
        "Norton Cancer Institute",
        "Louisville",
        "Kentucky",
        "4950 Norton Healthcare Blvd Suite 300, Louisville, KY 40241",
        38.2986,
        -85.5768,
        2200.0,
        4.0833,
        352.0,
        2,
    ),
    ClinicLocation(
        12,
        "Cancer Treatment Centers of America Chicago",
        "Zion",
        "Illinois",
        "2520 Elisha Ave, Zion, IL 60099",
        42.4461,
        -87.8329,
        2100.0,
        3.8333,
        314.0,
        2,
    ),
    ClinicLocation(
        13,
        "Massachusetts General Hospital Cancer Center",
        "Boston",
        "Massachusetts",
        "55 Fruit St, Boston, MA 02114",
        42.3632,
        -71.0689,
        3000.0,
        5.3333,
        506.0,
        2,
    ),
    ClinicLocation(
        14,
        "Memorial Sloan Kettering Cancer Center",
        "New York",
        "New York",
        "1275 York Ave, New York, NY 10065",
        40.7644,
        -73.9566,
        2900.0,
        5.1667,
        625.0,
        2,
    ),
    ClinicLocation(
        15,
        "Penn State Cancer Institute",
        "Hershey",
        "Pennsylvania",
        "400 University Dr, Hershey, PA 17033",
        40.2645,
        -76.6740,
        2700.0,
        6.6667,
        672.0,
        2,
    ),
    ClinicLocation(
        16,
        "MedStar Georgetown University Hospital",
        "Washington",
        "DC",
        "3800 Reservoir Rd NW, Washington, DC 20007",
        38.9121,
        -77.0753,
        2700.0,
        4.75,
        497.0,
        2,
    ),
    ClinicLocation(
        17,
        "Levine Cancer Institute",
        "Charlotte",
        "North Carolina",
        "1021 Morehead Medical Dr, Charlotte, NC 28204",
        35.2121,
        -80.8392,
        2500.0,
        4.5,
        428.0,
        2,
    ),
    ClinicLocation(
        18,
        "Winship Cancer Institute of Emory University",
        "Atlanta",
        "Georgia",
        "1365 E Clifton Rd NE Building C, Atlanta, GA 30322",
        33.7926,
        -84.3219,
        2200.0,
        4.0,
        377.0,
        2,
    ),
    ClinicLocation(
        19,
        "Mayo Clinic Florida",
        "Jacksonville",
        "Florida",
        "4500 San Pablo Rd S, Jacksonville, FL 32224",
        30.2641,
        -81.4393,
        2500.0,
        4.4167,
        588.0,
        2,
    ),
    ClinicLocation(
        20,
        "Miami Cancer Institute",
        "Miami",
        "Florida",
        "8900 N Kendall Dr, Miami, FL 33176",
        25.6863,
        -80.3407,
        2800.0,
        4.6667,
        570.0,
        2,
    ),
)


BB_20_CLINIC_COORDINATES: tuple[Coordinate, ...] = tuple(
    (clinic.latitude, clinic.longitude) for clinic in BB_20_CLINIC_LOCATIONS
)


def normalize_coordinates(
    coordinates: Sequence[Any] | None,
    expected_length: int | None = None,
) -> tuple[Coordinate, ...]:
    """Normalize coordinate config values into ``(latitude, longitude)`` pairs."""

    if coordinates is None:
        return ()
    normalized: list[Coordinate] = []
    for coordinate in coordinates:
        if isinstance(coordinate, Mapping):
            latitude = float(coordinate["latitude"])
            longitude = float(coordinate["longitude"])
        else:
            latitude = float(coordinate[0])
            longitude = float(coordinate[1])
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
        normalized.append((latitude, longitude))
    if expected_length is not None and len(normalized) != int(expected_length):
        raise ValueError(
            f"Expected {expected_length} clinic coordinates, got {len(normalized)}"
        )
    return tuple(normalized)


def haversine_miles(origin: Coordinate, destination: Coordinate) -> float:
    """Return great-circle distance between two latitude/longitude pairs."""

    lat1, lon1 = origin
    lat2, lon2 = destination
    radius_miles = 3958.7613
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    rlat1 = radians(lat1)
    rlat2 = radians(lat2)
    a = sin(dlat / 2.0) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2.0) ** 2
    return 2.0 * radius_miles * asin(sqrt(a))


def geographic_distance_matrix(coordinates: Sequence[Any]) -> tuple[tuple[float, ...], ...]:
    """Build a symmetric great-circle distance matrix in miles."""

    normalized = normalize_coordinates(coordinates)
    rows: list[tuple[float, ...]] = []
    for i, origin in enumerate(normalized):
        row = []
        for j, destination in enumerate(normalized):
            row.append(0.0 if i == j else haversine_miles(origin, destination))
        rows.append(tuple(row))
    return tuple(rows)


def geographic_transfer_time_matrix(
    coordinates: Sequence[Any],
    speed_mph: float = 500.0,
    fixed_handling_hours: float = 0.5,
) -> tuple[tuple[float, ...], ...]:
    """Estimate continuous transfer time in hours from great-circle distance.

    This is intentionally separate from ``transfer_lead_time`` in the simulator:
    the latter is an integer number of decision epochs, while this helper gives
    sub-epoch cold-chain/transport time for geography-aware costs and reporting.
    """

    speed = float(speed_mph)
    handling = float(fixed_handling_hours)
    if speed <= 0.0:
        raise ValueError("speed_mph must be positive")
    if handling < 0.0:
        raise ValueError("fixed_handling_hours must be nonnegative")

    distances = geographic_distance_matrix(coordinates)
    rows: list[tuple[float, ...]] = []
    for i, row in enumerate(distances):
        times = []
        for j, distance in enumerate(row):
            times.append(0.0 if i == j else handling + float(distance) / speed)
        rows.append(tuple(times))
    return tuple(rows)


def geographic_knn_edges(coordinates: Sequence[Any], k: int = 3) -> tuple[Edge, ...]:
    """Return a symmetric k-nearest-neighbor graph from coordinates."""

    normalized = normalize_coordinates(coordinates)
    n = len(normalized)
    if n < 2 or k < 1:
        return ()
    if k >= n:
        return complete_undirected_edges(n)

    edges: set[Edge] = set()
    for i, origin in enumerate(normalized):
        ranked = sorted(
            (
                (haversine_miles(origin, destination), j)
                for j, destination in enumerate(normalized)
                if j != i
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _distance, j in ranked[:k]:
            edges.add((min(i, j), max(i, j)))
    return tuple(sorted(edges))
