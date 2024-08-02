import pandas as pd
from sklearn.cluster import KMeans
from haversine import haversine
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import random
import string


def cluster_geo_data(file_path, n_clusters, use_haversine=False):
  """
  Clusters geographic data based on longitude and latitude.

  Args:
    file_path: Path to the CSV file.
    n_clusters: Number of desired clusters.
    use_haversine: Whether to use Haversine distance for clustering.

  Returns:
    A pandas DataFrame with an added 'cluster' column.
  """

  data = pd.read_csv(file_path)
  coordinates = data[['Longitude', 'Latitude']]

  if not use_haversine:
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(coordinates)
    data['cluster'] = kmeans.labels_
  else:
    # Haversine distance based clustering
    def haversine_distance(coord1, coord2):
      return haversine(coord1, coord2)

    distances = []
    for i in range(len(coordinates)):
      row = []
      for j in range(len(coordinates)):
        row.append(haversine_distance(coordinates.iloc[i], coordinates.iloc[j]))
      distances.append(row)

    distances = np.array(distances)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
    clustering.fit(distances)

    data['cluster'] = clustering.labels_

  return data


def map_clusters_to_letters(file_path):
  """
  Maps cluster numbers to random letter strings.

  Args:
    file_path: Path to the CSV file with cluster data.

  Returns:
    A pandas DataFrame with mapped cluster values.
  """

  data = pd.read_csv(file_path)

  # Create a mapping from cluster number to random string
  cluster_map = {}
  for cluster in data['cluster'].unique():
    random_string = ''.join(random.choice(string.ascii_letters) for _ in range(5))
    cluster_map[cluster] = random_string

  # Map cluster numbers to strings
  data['cluster_string'] = data['cluster'].map(cluster_map)

  return data


def drop_cluster_column(file_path):
  """Reads a CSV file and drops the 'cluster' column.

  Args:
    file_path: Path to the CSV file.

  Returns:
    A pandas DataFrame without the 'cluster' column.
  """

  data = pd.read_csv(file_path)
  data = data.drop('cluster', axis=1)
  return data


def add_cluster_string_count(file_path):
  """Adds a new column with cluster_string counts to each row.

  Args:
    file_path: Path to the CSV file.

  Returns:
    A pandas DataFrame with an added 'cluster_string_count' column.
  """

  df = pd.read_csv(file_path)

  # Calculate the value counts of 'cluster_string'
  cluster_string_counts = df['cluster_string'].value_counts().to_dict()

  # Create a new column with the corresponding counts
  df['cluster_string_count'] = df['cluster_string'].map(cluster_string_counts)

  return df


def add_cluster_string_index(file_path):
  """Adds a new column with index within cluster_string.

  Args:
    file_path: Path to the CSV file.

  Returns:
    A pandas DataFrame with added 'cluster_string_index' column.
  """

  df = pd.read_csv(file_path)

  # Group by 'cluster_string' and assign a cumulative count
  df['cluster_string_index'] = df.groupby('cluster_string').cumcount() + 1

  return df





# Example usage:
file_path = 'cal_dataframe.csv'
n_clusters = 75 # a reference number, there are 58 counties in California.
use_haversine = False  # Set to True if you want to use Haversine distance

data_with_clusters = cluster_geo_data(file_path, n_clusters, use_haversine)
data_with_clusters.to_csv('cal_dataframe_area_clusters.csv', index=False)

mapped_data = map_clusters_to_letters('cal_dataframe_area_clusters.csv')
mapped_data.to_csv('cal_dataframe_area_clusters_in_letters.csv', index=False)

data_cleansed = drop_cluster_column('cal_dataframe_area_clusters_in_letters.csv')
data_cleansed.to_csv('cal_dataframe_area_clusters_in_letters.csv', index=False)

counted_data = add_cluster_string_count('cal_dataframe_area_clusters_in_letters.csv')
counted_data.to_csv('cal_dataframe_area_clusters_in_letters_counted.csv', index=False)

counted_indexed_data = add_cluster_string_index('cal_dataframe_area_clusters_in_letters_counted.csv')
counted_indexed_data.to_csv('cal_dataframe_area_clusters_in_letters_counted_indexed.csv', index=False)

