import csv
import numpy as np
import pandas as pd
from utils_ACSIncome import generate_intervals
import folktables
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSHealthInsurance

WAOB = {
    1.0: 1,
    2.0: 0,
    3.0: 0,
    4.0: 0,
    5.0: 0,
    6.0: 0,
    7.0: 0,
    8.0: 0,
}


COW = {
    1.0: 1.0,
    2.0: 1.0,
    3.0: 1.0,
    4.0: 1.0,
    5.0: 1.0,
    6.0: 1.0,
    7.0: 1.0,
    8.0: 1.0,
    9.0: 0.0,
}

SCHL = {
    1.0: 0,
    2.0: 0,
    3.0: 0,
    4.0: 0,
    5.0: 0,
    6.0: 0,
    7.0: 0,
    8.0: 0,
    9.0: 0,
    10.0: 0,
    11.0: 0,
    12.0: 0,
    13.0: 0,
    14.0: 0,
    15.0: 0,
    16.0: 0,
    17.0: 0,
    18.0: 0,
    19.0: 0,
    20.0: 1,
    21.0: 1,
    22.0: 1,
    23.0: 1,
    24.0: 1,
}

MAR = {
    1: 1.0,  # Married
    2: 0.0,  # Widowed
    3: 0.0,  # Divorced
    4: 0.0,  # Separated
    5: 0.0,  # Never married or under 15 years old
}


RAC1P = {
    1.0: 1.0,  # White
    2.0: 0.0,  # Black or African American
    3.0: 0.0,  # American Indian or Alaska Native
    4.0: 0.0,  # Chinese
    5.0: 0.0,  # Japanese
    6.0: 0.0,  # Other Asian or Pacific Islander
    7.0: 0.0,  # Other race, including multiracial
}
wk_max = 99 
WKHP = {
    0: pd.Series(range(wk_max + 1)),
    1: generate_intervals(range(wk_max + 1), 0, 100, 5),
    2: generate_intervals(range(wk_max + 1), 0, 100, 25),
    3: generate_intervals(range(wk_max + 1), 0, 100, 50),
    4: np.array(["*"] * (wk_max + 1)),  # Ensure the size of the array is an integer
}

def write_hierachie(attr,csv_file) :
    
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        for status, binary_value in attr.items():
            writer.writerow([status, binary_value, "*"])

    print(f"Hierarchies saved to {csv_file}.")



def map_occp_to_group(occp_code):
    """
    Map an OCCP code to its corresponding occupation group based on the specified ranges.
    
    Args:
    occp_code (str): A 4-digit OCCP code (as a string).
    
    Returns:
    str: The name of the occupation group.
    """
    occp_code_int = int(occp_code)  # Convert OCCP code to integer for range comparisons
    
    # Define the mapping according to the given ranges
    if 10 <= occp_code_int <= 420:
        return 'Management, Business, Science, and Arts Occupations'
    elif 500 <= occp_code_int <= 730:
        return 'Business Operations Specialists'
    elif 800 <= occp_code_int <= 950:
        return 'Financial Specialists'
    elif 1000 <= occp_code_int <= 1240:
        return 'Computer and Mathematical Occupations'
    elif 1300 <= occp_code_int <= 1560:
        return 'Architecture and Engineering Occupations'
    elif 1600 <= occp_code_int <= 1965:
        return 'Life, Physical, and Social Science Occupations'
    elif 2000 <= occp_code_int <= 2060:
        return 'Community and Social Services Occupations'
    elif 2100 <= occp_code_int <= 2150: 
        return 'Legal Occupations'
    elif 2200 <= occp_code_int <= 2550:
        return 'Education, Training, and Library Occupations' 
    elif 2600 <= occp_code_int <=2920:
        return 'Arts, Design, Entertainment, Sports, and Media Occupations'
    elif 3000 <= occp_code_int <= 3550 : 
        return 'Healthcare Practitioners and Technical Occupations'
    elif 3600 <= occp_code_int <= 3650:
        return 'Healthcare Support Occupations'
    elif 3700 <= occp_code_int <= 3960:
        return 'Protective Service Occupations'
    elif 4000 <= occp_code_int <= 4160:
        return 'Food Preparation and Serving Occupations'
    elif 4200 <= occp_code_int <= 4255:
        return 'Building and Grounds Cleaning and Maintenance Occupations'
    elif 4300 <= occp_code_int <= 4655:
        return 'Personal Care and Service Occupations'
    elif 4700 <= occp_code_int <= 4965:
        return 'Sales and Related Occupations'
    elif 5000 <= occp_code_int <= 5940 : 
        return 'Office and Administrative Support Occupations'
    elif 6000 <= occp_code_int <= 6130 : 
        return 'Farming, Fishing, and Forestry Occupations'
    elif 6200 <= occp_code_int <= 6930 : 
        return 'Extraction Workers'
    elif 7000 <= occp_code_int <= 7610 :
        return 'Installation, Maintenance, and Repair Workers'
    elif 7700 <= occp_code_int <= 8990 :
        return 'Production Occupations'
    elif 9000 <= occp_code_int <= 9760 :
        return 'Transportation and Material Moving Occupations'
    elif 9800 <= occp_code_int <= 9830 :
        return 'Military Specific Occupations'
    else:
        return 'Unemployed'  # For codes outside the defined ranges








def map_occp_to_group_int(occp_code):
    """
    Map an OCCP code to its corresponding occupation group based on the specified ranges.
    
    Args:
    occp_code (str): A 4-digit OCCP code (as a string).
    
    Returns:
    str: The name of the occupation group.
    """
    occp_code_int = int(occp_code)  # Convert OCCP code to integer for range comparisons
    
    # Define the mapping according to the given ranges
    if 10 <= occp_code_int <= 420:
        return 1.0
    elif 500 <= occp_code_int <= 730:
        return 2.0
    elif 800 <= occp_code_int <= 950:
        return 3.0
    elif 1000 <= occp_code_int <= 1240:
        return 4.0
    elif 1300 <= occp_code_int <= 1560:
        return 5.0
    elif 1600 <= occp_code_int <= 1965:
        return 6.0
    elif 2000 <= occp_code_int <= 2060:
        return 7.0
    elif 2100 <= occp_code_int <= 2150: 
        return 8.0
    elif 2200 <= occp_code_int <= 2550:
        return 9.0
    elif 2600 <= occp_code_int <=2920:
        return 10.0
    elif 3000 <= occp_code_int <= 3550 : 
        return 11.0
    elif 3600 <= occp_code_int <= 3650:
        return 12.0
    elif 3700 <= occp_code_int <= 3960:
        return 13.0
    elif 4000 <= occp_code_int <= 4160:
        return 14.0
    elif 4200 <= occp_code_int <= 4255:
        return 15.0
    elif 4300 <= occp_code_int <= 4655:
        return 16.0
    elif 4700 <= occp_code_int <= 4965:
        return 17.0
    elif 5000 <= occp_code_int <= 5940 : 
        return 18.0
    elif 6000 <= occp_code_int <= 6130 : 
        return 19.0
    elif 6200 <= occp_code_int <= 6930 : 
        return 20.0
    elif 7000 <= occp_code_int <= 7610 :
        return 21.0
    elif 7700 <= occp_code_int <= 8990 :
        return 22.0
    elif 9000 <= occp_code_int <= 9760 :
        return 23.0
    elif 9800 <= occp_code_int <= 9830 :
        return 24.0
    else:
        return 0.0  # For codes outside the defined ranges






def map_pobp_to_region(pobp_code):
    """
    Map a POBP code to its corresponding region based on specified ranges.
    
    Args:
    pobp_code (int): A numeric POBP code.
    
    Returns:
    str: The name of the region or category.
    """
    # Define the mapping according to the ranges
    if 1 <= pobp_code <= 50:
        return 'United States (by State)'
    elif 51 <= pobp_code <= 56:
        return 'U.S. Territories'
    elif 60 <= pobp_code <= 100:
        return 'North America (Outside U.S.)'
    elif 101 <= pobp_code <= 200:
        return 'Central and South America'
    elif 201 <= pobp_code <= 300:
        return 'Europe'
    elif 301 <= pobp_code <= 400:
        return 'Asia'
    elif 401 <= pobp_code <= 500:
        return 'Africa'
    elif 501 <= pobp_code <= 600:
        return 'Oceania'
    elif 601 <= pobp_code <= 700:
        return 'Other/Unknown'
    else:
        return 'Invalid or Unspecified Code'



def map_pobp_to_region_int(pobp_code):
    """
    Map a POBP code to its corresponding region based on specified ranges.
    
    Args:
    pobp_code (int): A numeric POBP code.
    
    Returns:
    str: The name of the region or category.
    """
    # Define the mapping according to the ranges
    if 1 <= pobp_code <= 50:
        return 1.0
    elif 51 <= pobp_code <= 56:
        return 2.0
    elif 60 <= pobp_code <= 100:
        return 3.0
    elif 101 <= pobp_code <= 200:
        return 4.0
    elif 201 <= pobp_code <= 300:
        return 5.0
    elif 301 <= pobp_code <= 400:
        return 6.0
    elif 401 <= pobp_code <= 500:
        return 7.0
    elif 501 <= pobp_code <= 600:
        return 8.0
    elif 601 <= pobp_code <= 700:
        return 0.0
    else:
        return 0.0


def map_relp_to_category(relp_code):
    """
    Map RELP codes to broader relationship categories.
    
    Args:
    relp_code (int): RELP code (0–17).
    
    Returns:
    str: The category of the relationship.
    """
    # Define the categorization based on relationship types
    if relp_code == 0:
        return 'Reference Person'
    elif relp_code in [1, 2, 3, 4]:
        return 'Immediate Family'
    elif relp_code in [5, 6, 7, 8, 9, 10]:
        return 'Extended Family'
    elif relp_code in [11, 12, 13, 14, 15]:
        return 'Non-Family Household Members'
    elif relp_code == 16:
        return 'Institutionalized Group Quarters'
    elif relp_code == 17:
        return 'Noninstitutionalized Group Quarters'
    else:
        return 'Unknown Relationship'
    

def map_relp_to_category_int(relp_code):
    """
    Map RELP codes to broader relationship categories.
    
    Args:
    relp_code (int): RELP code (0–17).
    
    Returns:
    str: The category of the relationship.
    """
    # Define the categorization based on relationship types
    if relp_code == 0:
        return 0
    elif relp_code in [1, 2, 3, 4]:
        return 1
    elif relp_code in [5, 6, 7, 8, 9, 10]:
        return 2
    elif relp_code in [11, 12, 13, 14, 15]:
        return 3
    elif relp_code == 16:
        return 4
    elif relp_code == 17:
        return 5
    else:
        return 0



write_hierachie(COW,csv_file="hierarchies/ACSIncome/COW.csv")
write_hierachie(SCHL,csv_file="hierarchies/ACSIncome/SCHL.csv")
write_hierachie(MAR,csv_file="hierarchies/ACSIncome/MAR.csv")
write_hierachie(RAC1P,csv_file="hierarchies/ACSIncome/RAC1P.csv")
write_hierachie(WKHP,csv_file="hierarchies/ACSIncome/WKHP.csv")
write_hierachie(WAOB,csv_file="hierarchies/ACSIncome/WAOB.csv")



data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(download=True)
features, label, _ = ACSIncome.df_to_pandas(acs_data)
occp_vals= features['OCCP'].unique()
pobp_vals= features['POBP'].unique()
relp_vals= features['RELP'].unique()

print("columns in ACSIncome : ", features['OCCP'].unique())

OCCP = {float(code): map_occp_to_group_int(code) for code in sorted(occp_vals)}
POBP = {float(code): map_pobp_to_region_int(code) for code in sorted(pobp_vals)}
RELP = {float(code): map_relp_to_category_int(code) for code in sorted(relp_vals)}


write_hierachie(OCCP,csv_file="hierarchies/ACSIncome/OCCP.csv")
write_hierachie(POBP,csv_file="hierarchies/ACSIncome/POBP.csv")
write_hierachie(RELP,csv_file="hierarchies/ACSIncome/RELP.csv")
