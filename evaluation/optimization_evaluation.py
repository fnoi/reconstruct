import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_data_from_file(file_path):
    """
    Extracts iteration numbers and corresponding cost values from a log file.

    Args:
        file_path (str): Path to the log file.

    Returns:
        dict: A dictionary where keys are iteration numbers and values are cost values.
    """
    with open(file_path, 'r') as file:
        content = file.read()
        content = content.replace('\r\n', '\n')  # Normalize line endings

    # check if content is empty
    if not content:
        raise ValueError(f"File is empty: {file_path}")
    pattern = r"(?:New best for swarm at iteration|Best after iteration) (\d+): \[([-+\de.\s\n]+)\] ([\d.e-]+)"

    matches = re.findall(pattern, content, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"No matches found in the file: {file_path}")

    cost_data = {}
    for match in matches:
        iteration = int(match[0])
        cost_value = float(match[2])
        cost_data[iteration] = cost_value

    return cost_data


def process_log_files(directory):
    """
    Processes all log files in the specified directory, extracting iteration numbers and cost values,
    and combining them into a single DataFrame with filenames as columns.

    Args:
        directory (str): Path to the directory containing the log files.

    Returns:
        pd.DataFrame: A DataFrame with iteration numbers as index and filenames as columns.
    """
    all_data = {}
    all_data_raw = {}

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            cost_data = extract_data_from_file(file_path)
            all_data_raw[filename] = cost_data
            # normalize data
            cost_data_array = np.array(list(cost_data.values()))
            c_max = np.max(cost_data_array)
            c_min = np.min(cost_data_array)
            c_range = c_max - c_min
            cost_data = {k: (v - c_min) / c_range for k, v in cost_data.items()}



            all_data[filename] = cost_data

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(all_data)
    df_raw = pd.DataFrame(all_data_raw)
    df.index.name = 'Iteration'
    df_raw.index.name = 'Iteration'

    return df, df_raw


def plot_min_cost_vs_iteration(df):
    """
    Plots the minimum cost versus iteration from the processed DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing iteration and cost data from multiple files.
    """
    df_min = df.min(axis=1)

    plt.figure(figsize=(5, 3))
    # for each of the columns where name starts with 'output'
    for column in df.columns:
        if column.startswith('output'):
            plt.plot(df.index, df[column], alpha=0.5, label=column)
    # plt.plot(df.index, df_min, color='blue', label='Min')
    plt.xlabel('Iteration')

    x_ticks = np.arange(0, 101, 10)
    plt.xticks(x_ticks)

    plt.ylabel('Cost')
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    # save to file
    plt.savefig('output.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Set the path to your log directory
    log_directory = '/Users/fnoic/PycharmProjects/reconstruct/experiment_log/'

    # Process the log files and create a DataFrame
    result_df, result_df_raw = process_log_files(log_directory)

    # # Print the resulting DataFrame
    # print(result_df)

    # Plot the minimum cost versus iteration
    plot_min_cost_vs_iteration(result_df)
    plot_min_cost_vs_iteration(result_df_raw)

