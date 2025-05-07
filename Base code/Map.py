# Required imports
import numpy as np
from Location import Location
from Boundaries import Boundaries
from Radar import Radar
from tqdm import    tqdm

# Constant that avoids setting cells to have an associated cost of zero
EPSILON = 1e-4

class Map:
    """ Class that models the map for the simulation """
    def __init__(self, 
                 boundaries: Boundaries,
                 height:     np.int32, 
                 width:      np.int32, 
                 radars:     np.array=None):
        self.boundaries = boundaries        # Boundaries of the map
        self.height     = height            # Number of coordinates in the y-axis
        self.width      = width             # Number of coordinates int the x-axis
        self.radars     = radars            # List containing the radars (objects)

    def generate_radars(self, n_radars: np.int32) -> None:
        """ Generates n-radars randomly and inserts them into the radars list """
        # Select random coordinates inside the boundaries of the map
        lat_range = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)
        rand_lats = np.random.choice(a=lat_range, size=n_radars, replace=False)
        rand_lons = np.random.choice(a=lon_range, size=n_radars, replace=False)
        self.radars = []        # Initialize 'radars' as an empty list

        # Loop for each radar that must be generated
        for i in range(n_radars):
            # Create a new radar
            new_radar = Radar(location=Location(latitude=rand_lats[i], longitude=rand_lons[i]),
                              transmission_power=np.random.uniform(low=1, high=1000000),
                              antenna_gain=np.random.uniform(low=10, high=50),
                              wavelength=np.random.uniform(low=0.001, high=10.0),
                              cross_section=np.random.uniform(low=0.1, high=10.0),
                              minimum_signal=np.random.uniform(low=1e-10, high=1e-15),
                              total_loss=np.random.randint(low=1, high=10),
                              covariance=None)

            # Insert the new radar
            self.radars.append(new_radar)
        return
    
    def get_radars_locations_numpy(self) -> np.array:
        """ Returns an array with the coordiantes (lat, lon) of each radar registered in the map """
        locations = np.zeros(shape=(len(self.radars), 2), dtype=np.float32)
        for i in range(len(self.radars)):
            locations[i] = self.radars[i].location.to_numpy()
        return locations

    def compute_detection_map(self) -> np.array:
        """ Computes the detection map for each coordinate in the map (with all the radars) """
        # Create an empty detection map
        detection_map = np.zeros(shape=(self.height, self.width), dtype=np.float32)

        # Create arrays with the latitude and longitude values for each cell in the grid
        lat_values = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_values = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)

        # For each cell in the map
        for i in tqdm(range(self.height), desc="Computing detection map"):  # Using tqdm for progress bar
            for j in range(self.width):
                # Get the latitude and longitude of the current cell
                current_lat = lat_values[i]
                current_lon = lon_values[j]

                # Calculate the detection level for each radar and keep the maximum value
                max_detection = 0.0
                for radar in self.radars:
                    detection_level = radar.compute_detection_level(latitude=current_lat, longitude=current_lon)
                    max_detection = max(max_detection, detection_level)

                # Assign the maximum detection value to the cell
                detection_map[i, j] = max_detection

        # Scale the detection map using min-max scaling with epsilon
        if np.max(detection_map) - np.min(detection_map) > 0:  # Avoid division by zero
            detection_map = (detection_map - np.min(detection_map)) / (
                        np.max(detection_map) - np.min(detection_map)) * (1 - EPSILON) + EPSILON
        else:
            # If all values are the same, set them to epsilon
            detection_map.fill(EPSILON)

        return detection_map