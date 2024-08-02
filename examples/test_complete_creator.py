import pandas as pd

def create_new_csv(input_csv, output_csv, num_rows):
  """Creates a new CSV file with specified columns and values.

  Args:
    input_csv: Path to the input CSV file.
    output_csv: Path to the output CSV file.
    num_rows: Number of rows for the new CSV file.
  """

  # Read the input CSV to get column names
  df = pd.read_csv(input_csv)
  columns = df.columns

  # Create a new DataFrame with specified values
  data = {
      'cluster_string': ['rzDye'] * num_rows,
      'cluster_string_index': range(1, num_rows + 1),
      'cluster_string_count': [num_rows] * num_rows
  }
  new_df = pd.DataFrame(data, columns=columns)

  # Fill missing columns with NaN
  for col in columns:
      if col not in data:
          new_df[col] = None

  # Save the new DataFrame to a CSV file
  new_df.to_csv(output_csv, index=False)

# Example usage:
input_file = 'cal_dataframe_area_clusters_in_letters_counted_indexed.csv'
output_file = 'test_complete.csv'
num_rows = 56

create_new_csv(input_file, output_file, num_rows)
