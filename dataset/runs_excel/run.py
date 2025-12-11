from openpyxl import load_workbook

# Load your Excel file
file_path = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\runs_excel\run_002.xlsx"  # <-- replace with your Excel file path
wb = load_workbook(file_path)
ws = wb.active  # use specific sheet with wb['SheetName'] if needed

# Delete rows 2 to 6
ws.delete_rows(2, 5)  # 2 is the starting row, 5 is the number of rows to delete

# Save the changes
wb.save("your_file_modified.xlsx")
print("Rows 2-6 deleted and remaining rows lifted up successfully!")