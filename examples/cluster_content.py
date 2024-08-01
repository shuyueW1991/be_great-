import pandas as pd
def group_and_concatenate(file_path):
  """Groups data by 'cluster_string' and concatenates rows.

  Args:
    file_path: Path to the CSV file.

  Returns:
    A pandas DataFrame with 'cluster_string' and 'content' columns.
  """

  df = pd.read_csv(file_path)

  # Assuming you want to concatenate all columns except 'cluster_string'
  columns_to_concatenate = df.columns.difference(['cluster_string'])

  def concatenate_rows(group):
    group = group.drop('cluster_string', axis=1)  # Drop 'cluster_string' for concatenation
    group = group.astype(str)  # Convert all columns to strings for concatenation
    group = group.apply(lambda row: '|'.join(row), axis=1)  # Concatenate each row
    return pd.Series({'cluster_string': group.name, 'content': '|'.join(group)})

  df_grouped = df.groupby('cluster_string').apply(concatenate_rows)
  df_grouped = df_grouped.reset_index(drop=True)
  return df_grouped


# Example usage:
file_path = 'cal_dataframe_area_clusters_in_letters.csv'
result_df = group_and_concatenate(file_path)
result_df.to_csv('cal_dataframe_cluster_content.csv')