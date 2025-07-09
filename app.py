from flask import Flask, request, jsonify
import pickle
import pandas as pd
import math
import re
from flask_cors import CORS

# Load your model and preprocessor
with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)
models = bundle['models']
preprocessor = bundle['preprocessor']

#feature extraction code to process user input

def parse_composition_series(composition_series):
    """
    Parse a pandas Series of chemical composition strings into dictionaries.
    """
    def _parse_single_composition(composition_string):
        """Helper function to parse a single composition string."""
        # Handle NaN or None values
        if pd.isna(composition_string) or composition_string is None:
            return {}

        # Pattern to match element symbol followed by optional number
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, str(composition_string))
        composition_dict = {}

        for element, amount in matches:
            # If no amount is specified, default to 1.0
            if amount == '':
                amount = 1.0
            else:
                try:
                    amount = float(amount)
                except ValueError:
                    # Skip invalid amounts
                    continue

            composition_dict[element] = amount
        return composition_dict
    return composition_series.apply(_parse_single_composition)

def convert_to_percentages_key_elements(composition_dict, key_elements):
    """
    Convert a composition dictionary to percentages for key elements only.
    """
    # Handle empty dictionaries
    if not composition_dict:
        return {element: 0.0 for element in key_elements}

    # Calculate total for normalization
    total = sum(composition_dict.values())
    if total == 0:
        return {element: 0.0 for element in key_elements}

    # Only include key elements in the result
    result = {}
    for element in key_elements:
        if element in composition_dict:
            result[element] = round((composition_dict[element] / total) * 100, 2)
        else:
            result[element] = 0.0

    return result

def convert_series_to_percentages_key_elements(composition_series, key_elements):
    """
    Convert a pandas Series of composition dictionaries to percentages for key elements only.
    """
    return composition_series.apply(lambda comp: convert_to_percentages_key_elements(comp, key_elements))

def add_key_element_columns_to_dataframe(data, percentage_compositions):
    """
    Add key element percentage columns to the dataframe.
    """
    # Create DataFrame with key elements
    element_data = []
    for composition_dict in percentage_compositions:
        element_data.append(composition_dict)

    element_df = pd.DataFrame(element_data)

    # Add percentage suffix to column names for clarity
    element_df.columns = [f'{element}_pct' for element in element_df.columns]

    # Concatenate with original dataframe
    result_df = pd.concat([data, element_df], axis=1)

    return result_df, list(element_df.columns)

def classify_crystal_system(
    a, b, c, alpha, beta, gamma,
    length_tol=0.00001, angle_tol=0.1
):
    """
    defines the crystal system based on the lattice parameters
    """
    # Convert all to float
    a, b, c, alpha, beta, gamma = map(float, [a, b, c, alpha, beta, gamma])

    # Separate helpers for lengths and angles
    def eq_length(x, y): return math.isclose(x, y, abs_tol=length_tol)
    def eq_angle(x, y):  return math.isclose(x, y, abs_tol=angle_tol)

    # Cubic: all sides equal, all angles 90
    if eq_length(a, b) and eq_length(b, c) and \
       eq_angle(alpha, 90) and eq_angle(beta, 90) and eq_angle(gamma, 90):
        return "cubic"
    # Tetragonal: a = b ≠ c, all angles 90
    elif eq_length(a, b) and not eq_length(b, c) and \
         eq_angle(alpha, 90) and eq_angle(beta, 90) and eq_angle(gamma, 90):
        return "tetragonal"
    # Orthorhombic: a ≠ b ≠ c, all angles 90
    elif not eq_length(a, b) and not eq_length(b, c) and not eq_length(a, c) and \
         eq_angle(alpha, 90) and eq_angle(beta, 90) and eq_angle(gamma, 90):
        return "orthorhombic"
    # Hexagonal: a = b ≠ c, alpha = beta = 90, gamma = 120
    elif eq_length(a, b) and not eq_length(b, c) and \
         eq_angle(alpha, 90) and eq_angle(beta, 90) and eq_angle(gamma, 120):
        return "hexagonal"
    # Trigonal (Rhombohedral): a = b = c, alpha = beta = gamma ≠ 90
    elif eq_length(a, b) and eq_length(b, c) and \
         eq_angle(alpha, beta) and eq_angle(beta, gamma) and not eq_angle(alpha, 90):
        return "trigonal"
    # Monoclinic: a ≠ b ≠ c, alpha = gamma = 90, beta ≠ 90
    elif not eq_length(a, b) and not eq_length(b, c) and not eq_length(a, c) and \
         eq_angle(alpha, 90) and not eq_angle(beta, 90) and eq_angle(gamma, 90):
        return "monoclinic"
    # Triclinic: a ≠ b ≠ c, all angles ≠ 90
    elif not eq_length(a, b) and not eq_length(b, c) and not eq_length(a, c) and \
         not eq_angle(alpha, 90) and not eq_angle(beta, 90) and not eq_angle(gamma, 90):
        return "triclinic"
    else:
        return "unknown"

def calculate_unit_cell_volume(a, b, c, alpha, beta, gamma):
  """
  Calculate the volume of a unit cell with right angles (α=β=γ=90°).

  Parameters:
        a (float): Length of unit cell edge a
        b (float): Length of unit cell edge b
        c (float): Length of unit cell edge c

    Returns:
        float: Volume of the unit cell
  """
  a, b, c, alpha, beta, gamma = map(float, [a, b , c, alpha, beta, gamma]) #covnverting to floats so i can do math
  alpha, beta, gamma = map(math.radians, [alpha, beta, gamma]) #converting to radians for math.cos

  #standard equation for calculating volume, this is because not all crystal structures have a unit cell with angles 90 degrees. this is volume of a parallelepiped, which is made of 3 vectors (a b and c)
  unit_cell_volume = a * b * c * math.sqrt(1-math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2 + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma))

  return unit_cell_volume

def get_li_concentration_per_formula_unit(Composition):
  """
  Extracts the number of Li atoms per formula unit from a chemical formula string.
  If no subscript is present after 'Li', returns 1.
  If 'Li' is not found, returns 0.
  """
  match = re.search(r'Li([0-9\.]*)', Composition) #this is looking for a pattern of Li followed by any number 0 to 9, where the star means that this pattern must occur 0 or more times. ie it will still work if the Li doesn't have a subscript
  if match:
    num = match.group(1) #the first thing in the re.search is the first group. in this case the second group (index 1) is the number
    if num == '':
      return 1 #if no subscript, there is only 1 Li atom/unit
    else:
      return float(num)
  return 0 #no Li found


def extract_features_from_composition(composition, temperature=25.0):
    # --- Defaults for structural features (could be improved with user input) ---
    a = b = c = alpha = beta = gamma = 1.0  # Replace with real values if available
    space_group_number = 1  # Replace with real value if available

    # --- Key elements list ---
    key_elements = [
        'Li', 'F', 'O', 'S', 'C', 'N', 'Ti', 'Zr', 'Al', 'P', 'Si', 'Ge', 'La', 'Cl', 'Br', 'I'
    ]

    # --- Parse composition and get element percentages ---
    comp_dict = parse_composition_series(pd.Series([composition]))[0]
    pct_dict = convert_to_percentages_key_elements(comp_dict, key_elements)

    # --- Feature engineering ---
    li_conc = get_li_concentration_per_formula_unit(composition)
    unit_cell_volume = calculate_unit_cell_volume(a, b, c, alpha, beta, gamma)
    crystal_system = classify_crystal_system(a, b, c, alpha, beta, gamma)

    # --- Physics-based features ---
    # Ionic radii and electronegativity values
    ionic_radii = {
        'Li': 0.76, 'Na': 1.02, 'K': 1.38, 'Rb': 1.52, 'Cs': 1.67,
        'F': 1.33, 'Cl': 1.81, 'Br': 1.96, 'I': 2.20,
        'O': 1.40, 'S': 1.84, 'Se': 1.98, 'Te': 2.21,
        'Al': 0.535, 'Ti': 0.605, 'Zr': 0.72, 'P': 0.44, 'Si': 0.40
    }
    electronegativity = {
        'Li': 0.98, 'F': 3.98, 'O': 3.44, 'S': 2.58, 'Cl': 3.16,
        'P': 2.19, 'Si': 1.90, 'Al': 1.61, 'Ti': 1.54, 'Zr': 1.33
    }

    # Weighted ionic radius
    total_weighted_radius = 0
    total_atoms = 0
    for element in ionic_radii.keys():
        pct = pct_dict.get(element, 0.0)
        if pct > 0:
            weight = pct / 100.0
            total_weighted_radius += ionic_radii[element] * weight
            total_atoms += weight
    weighted_ionic_radius = total_weighted_radius / total_atoms if total_atoms > 0 else 0

    # Electronegativity range
    electronegativities = [electronegativity[e] for e in electronegativity if pct_dict.get(e, 0.0) > 0]
    if len(electronegativities) >= 2:
        electronegativity_range = max(electronegativities) - min(electronegativities)
    else:
        electronegativity_range = 0

    # Element count
    element_count = sum(1 for el in key_elements if pct_dict.get(el, 0.0) > 0)

    # --- Build feature dict ---
    features = {
        'Space group number': space_group_number,
        'a': a,
        'b': b,
        'c': c,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'temperature': temperature,
        '[Li+]/formula unit': li_conc,
        'Unit cell volume (Angstroms cubed)': unit_cell_volume,
        'Chemical Family': 'unknown',          # Or set a better default
        'Structural Family': 'unknown',        # Or set a better default
        'Crystal System': crystal_system,
        'weighted_ionic_radius': weighted_ionic_radius,
        'b_a_ratio': b / a if a else 0,
        'c_a_ratio': c / a if a else 0,
        'c_b_ratio': c / b if b else 0,
        'li_density': li_conc / unit_cell_volume if unit_cell_volume else 0,
        'inverse_temperature': 1000.0 / (temperature + 273.15),
        'electronegativity_range': electronegativity_range,
        'element_count': element_count
    }
    # Add all element percentages
    for el in key_elements:
        features[f"{el}_pct"] = pct_dict.get(el, 0.0)

    return features

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    composition = data.get('composition')
    # Your ML model prediction logic here
    prediction = "dummy_result"  # Replace with your model's prediction
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

