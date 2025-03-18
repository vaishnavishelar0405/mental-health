
# Path to the uploaded dataset
file_path = r"C:\Users\omkur\OneDrive\Desktop\Project\mental_health_data_final.csv"

# Count the number of lines (rows) in the dataset
with open(file_path, "r", encoding="utf-8") as file:
    num_lines = sum(1 for line in file)

# Return the number of rows (excluding header)
num_lines
print(num_lines)

