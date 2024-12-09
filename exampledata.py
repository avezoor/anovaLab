import pandas as pd
data1 = {
    'Test': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
    'Score': [80, 85, 90, 78, 82, 88, 90, 85, 87, 91, 70, 72, 68, 75, 69]
}
data2 = {
    'Test': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
    'Gender': ['Male', 'Male', 'Male', 'Female', 'Female', 'Female', 
                      'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 
                      'Male', 'Male', 'Male', 'Female', 'Female', 'Female'],
    'Score': [80, 85, 90, 75, 78, 82, 88, 92, 91, 85, 87, 90, 70, 72, 75, 68, 69, 70]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

with pd.ExcelWriter('onewaydata.xlsx', engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='One-Way ANOVA', index=False)
with pd.ExcelWriter('twowaydata.xlsx', engine='openpyxl') as writer:
    df2.to_excel(writer, sheet_name='Two-Way ANOVA', index=False)

print("Excel file has been successfully created.")
