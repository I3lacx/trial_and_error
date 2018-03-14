import gspread
from oauth2client.service_account import ServiceAccountCredentials

path = "E:\Programmieren\Python\learnPython\trial_and_error\metin2"
scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

sheet = client.open('Preise').sheet1

col = sheet.col_values(2)
row = sheet.row_values(1)
cellVal = sheet.cell(1,2).value
print(cellVal)
sheet.update_cell(1,2, 'Datum')
cellVal = sheet.cell(1,2).value
print(cellVal)

new_row = ["Banediger", "20.01", "10000"]
index = 2
sheet.insert_row(new_row, index)

sheet.delete_row(2)
