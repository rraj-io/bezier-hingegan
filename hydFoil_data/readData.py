import os
import glob
import numpy as np
import ast
from typing import List, Dict, Union


def read_numerical_data(file_path: str) -> np.ndarray:
    """
    Reads numerical data from a file. Each line is expected to be a sequence of numbers.

    Args:
    file_path (str): The path to the file to be read.

    Returns:
    np.ndarray: A NumPy array of the numerical data.
    """
    return np.loadtxt(file_path)


def read_dict_data(file_path: str) -> List[Dict[str, float]]:
    """
    Reads a file with dictionary-like content (e.g., {'tl': val, 'n': val, 'vl': val}).
    Each line in the file is a dictionary.

    Args:
    file_path (str): The path to the file to be read.

    Returns:
    List[Dict[str, float]]: A list of dictionaries for each row in the file.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                parsed_line = ast.literal_eval(line.strip())
                data.append(parsed_line)
            except Exception as e:
                print(f"Error parsing line: {line}, Error: {e}")
    return data


def read_all_data(datadir: str) -> Dict[str, Union[List[Dict[str, float]], np.ndarray]]:
    """
    Reads all required data (P, dH, eta, VCav, islandID, input, fitness) from multiple files
    in the specified directory and stores them in a dictionary.

    Args:
    datadir (str): Path to the directory containing the data files.

    Returns:
    Dict[str, Union[List[Dict[str, float]], np.ndarray]]: A dictionary with all the data concatenated.
    """
    data = {}

    def read_and_concatenate_dict(file_prefix: str) -> List[Dict[str, float]]:
        all_data = []
        for file in sorted(glob.glob(os.path.join(datadir, file_prefix + ".*"))):
            all_data.extend(read_dict_data(file))
        return all_data

    def read_and_concatenate_numeric(file_prefix: str) -> np.ndarray:
        all_data = []
        for file in sorted(glob.glob(os.path.join(datadir, file_prefix + ".*"))):
            file_data = read_numerical_data(file)
            all_data.append(file_data)
        return np.concatenate(all_data, axis=0)

    def read_and_concatenate_1d(file_prefix: str) -> np.ndarray:
        all_data = []
        for file in sorted(glob.glob(os.path.join(datadir, file_prefix + ".*"))):
            file_data = read_numerical_data(file).flatten()
            all_data.append(file_data)
        return np.hstack(all_data)

    data["F"] = read_and_concatenate_dict("F")
    data["dH"] = read_and_concatenate_dict("dH")
    data["eta"] = read_and_concatenate_dict("eta")
    # data['VCav'] = read_and_concatenate_dict('VCav')
    data["input"] = read_and_concatenate_numeric("objective")
    data["id"] = read_and_concatenate_1d("id")
    data["fitness"] = read_and_concatenate_numeric("fitness")

    return data


# def DoF():
#   return [
#       {'label': 'cV_ru_alpha_1_ex_0.0', 'min': -0.155, 'max': 0.025},
#        {'label': 'cV_ru_alpha_1_ex_0.5', 'min': -0.19, 'max': -0.01},
#        {'label': 'cV_ru_alpha_1_ex_1.0', 'min': -0.19, 'max': -0.01},
#        {'label': 'cV_ru_alpha_2_ex_0.0', 'min': -0.08, 'max': 0.1},
#        {'label': 'cV_ru_alpha_2_ex_0.5', 'min': -0.08, 'max': 0.1},
#        {'label': 'cV_ru_alpha_2_ex_1.0', 'min': -0.08, 'max': 0.07},
#        {'label': 'cV_ru_offsetM_ex_0.0', 'min': 1.0, 'max': 1.5},
#        {'label': 'cV_ru_offsetM_ex_0.5', 'min': 1.0, 'max': 1.5},
#        {'label': 'cV_ru_offsetM_ex_1.0', 'min': 1.0, 'max': 1.5},
#        {'label': 'cV_ru_ratio_0.0', 'min': 0.4, 'max': 0.6},
#        {'label': 'cV_ru_ratio_0.5', 'min': 0.4, 'max': 0.6},
#        {'label': 'cV_ru_ratio_1.0', 'min': 0.4, 'max': 0.6},
#        {'label': 'cV_ru_offsetPhiR_ex_0.0', 'min': -0.15, 'max': 0.15},
#        {'label': 'cV_ru_offsetPhiR_ex_0.5', 'min': -0.15, 'max': 0.15},
#        {'label': 'cV_ru_offsetPhiR_ex_1.0', 'min': -0.15, 'max': 0.15},
#        {'label': 'cV_ru_bladeLength_0.0', 'min': 0.4, 'max': 0.8},
#        {'label': 'cV_ru_bladeLength_0.5', 'min': 0.6, 'max': 1.0},
#        {'label': 'cV_ru_bladeLength_1.0', 'min': 0.8, 'max': 1.3},
#        {'label': 'cV_ru_t_le_a_0', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_t_le_a_0.5', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_t_le_a_1', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_t_mid_a_0', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_t_mid_a_0.5', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_t_mid_a_1', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_t_te_a_0', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_t_te_a_0.5', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_t_te_a_1', 'min': 0.005, 'max': 0.06},
#        {'label': 'cV_ru_u_mid_a_0', 'min': 0.4, 'max': 0.6},
#        {'label': 'cV_ru_u_mid_a_0.5', 'min': 0.4, 'max': 0.6},
#        {'label': 'cV_ru_u_mid_a_1', 'min': 0.4, 'max': 0.6}
#    ]


# Updating DoF() for HydroFoil Optimization
def DoF():
    return [
        {'label': 'alpha_1', 'min': 120.0, 'max': 180.0},
        {'label': 'alpha_2', 'min': 120.0, 'max': 180.0},
        {'label': 't_mid', 'min': 0.005, 'max': 0.06},
        {'label': 't_le', 'min': 0.005, 'max': 0.06},
        {'label': 't_te', 'min': 0.005, 'max': 0.06},
        {'label': 'ratio', 'min': 0.4, 'max': 0.6},
        {'label': 'deltaM', 'min': 1.0, 'max': 1.5},
        {'label': 'offM', 'min': 1.0, 'max': 1.5},
        {'label': 'bladeLength', 'min': 0.4, 'max': 0.80},
    ]

# Function to normalize the parameters


def normalize_all_data(data: np.ndarray) -> np.ndarray:
    """
    Normalizes data['input'] to a 0-1 range based on the min and max values in DoF.

    Args:
    data (np.ndarray): The array of original parameters, shape (n, 30).

    Returns:
    np.ndarray: Normalized array with values between 0 and 1, shape (n, 30).
    """
    parameter_ranges = DoF()
    normalized_data = np.zeros_like(data)
    for i, param in enumerate(parameter_ranges):
        min_val = param["min"]
        max_val = param["max"]
        if data.ndim == 1:
            normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
        else:
            normalized_data[:, i] = (data[:, i] - min_val) / (max_val - min_val)
    return normalized_data


# Function to denormalize the parameters


def denormalize_all_data(normalized_data: np.ndarray) -> np.ndarray:
    """
    Converts normalized 0-1 values back to original parameter ranges based on DoF.

    Args:
    normalized_data (np.ndarray): The normalized array with values between 0 and 1, shape (n, 30).

    Returns:
    np.ndarray: Denormalized array with original parameter ranges, shape (n, 30).
    """
    parameter_ranges = DoF()
    original_data = np.zeros_like(normalized_data)
    for i, param in enumerate(parameter_ranges):
        min_val = param["min"]
        max_val = param["max"]
        if normalized_data.ndim == 1:
            original_data[i] = normalized_data[i] * (max_val - min_val) + min_val
        else:
            original_data[:, i] = normalized_data[:, i] * (max_val - min_val) + min_val
        # original_data[:, i] = normalized_data[:, i] * \
        #     (max_val - min_val) + min_val
    return original_data


def normalize_parameters(individual: np.ndarray) -> np.ndarray:
    """
    Normalizes a single individual's parameters to 0-1 range based on DoF.

    Args:
    individual (np.ndarray): An array of 30 original parameters for a single individual.

    Returns:
    np.ndarray: A normalized array with values between 0 and 1, shape (30,).
    """
    parameter_ranges = DoF()
    normalized_individual = np.zeros_like(individual)
    for i, param in enumerate(parameter_ranges):
        min_val = param["min"]
        max_val = param["max"]
        normalized_individual[i] = (individual[i] - min_val) / (max_val - min_val)
    return normalized_individual


def denormalize_parameters(normalized_individual: np.ndarray) -> np.ndarray:
    """
    Converts a single normalized individual's parameters back to the original scale.

    Args:
    normalized_individual (np.ndarray): A normalized array with values between 0 and 1, shape (30,).

    Returns:
    np.ndarray: Denormalized array with original parameter ranges, shape (30,).
    """
    parameter_ranges = DoF()
    original_individual = np.zeros_like(normalized_individual)
    for i, param in enumerate(parameter_ranges):
        min_val = param["min"]
        max_val = param["max"]
        original_individual[i] = (
            normalized_individual[i] * (max_val - min_val) + min_val
        )
    return original_individual


# Place code to execute only when run as standalone script below
if __name__ == "__main__":
    # Example usage
    DATADIR = "./runData"  # You can replace this with any path you prefer
    data = read_all_data(DATADIR)
    print("Data has been read successfully.")
