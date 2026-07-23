import unittest

from src.graph.geography import (
    BB_20_CLINIC_COORDINATES,
    BB_20_CLINIC_LOCATIONS,
    geographic_knn_edges,
    geographic_transfer_time_matrix,
    haversine_miles,
    normalize_coordinates,
)


class GeographyTest(unittest.TestCase):
    def test_bb_clinic_location_metadata_is_complete(self):
        self.assertEqual(len(BB_20_CLINIC_LOCATIONS), 20)
        self.assertEqual(tuple(location.index for location in BB_20_CLINIC_LOCATIONS), tuple(range(1, 21)))
        self.assertEqual(len(BB_20_CLINIC_COORDINATES), 20)
        self.assertEqual(BB_20_CLINIC_LOCATIONS[0].name, "Seattle Cancer Care Alliance")
        self.assertEqual(BB_20_CLINIC_LOCATIONS[-1].name, "Miami Cancer Institute")
        self.assertEqual(BB_20_CLINIC_LOCATIONS[6].supplier, 1)
        self.assertEqual(BB_20_CLINIC_LOCATIONS[8].supplier, 2)

    def test_haversine_miles_is_reasonable_for_california_pair(self):
        los_angeles = (34.0485, -118.2577)
        san_diego = (32.7530, -117.1650)

        distance = haversine_miles(los_angeles, san_diego)

        self.assertGreater(distance, 100.0)
        self.assertLess(distance, 130.0)

    def test_transfer_time_matrix_is_continuous_hours(self):
        los_angeles = (34.0485, -118.2577)
        san_diego = (32.7530, -117.1650)

        matrix = geographic_transfer_time_matrix(
            (los_angeles, san_diego),
            speed_mph=500.0,
            fixed_handling_hours=0.5,
        )

        self.assertEqual(matrix[0][0], 0.0)
        self.assertEqual(matrix[1][1], 0.0)
        self.assertGreater(matrix[0][1], 0.5)
        self.assertLess(matrix[0][1], 1.0)
        self.assertEqual(matrix[0][1], matrix[1][0])

    def test_geographic_knn_edges_uses_nearest_coordinates(self):
        coordinates = ((0.0, 0.0), (0.0, 1.0), (20.0, 20.0), (20.0, 21.0))

        edges = geographic_knn_edges(coordinates, k=1)

        self.assertEqual(edges, ((0, 1), (2, 3)))

    def test_normalize_coordinates_accepts_mapping_config_values(self):
        coordinates = [{"latitude": 1.0, "longitude": 2.0}]

        self.assertEqual(normalize_coordinates(coordinates, expected_length=1), ((1.0, 2.0),))


if __name__ == "__main__":
    unittest.main()
