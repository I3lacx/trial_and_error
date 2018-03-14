import pygsheets

gc = pygsheets.authorize()

print("heyho")
sheet = gc.open('Preise').sheet1
sheet.update_cell('B1', 'Datum' )

new_row = ["Banediger", "20.01", "10000"]
sheet.insert_rows(row=5,number=1,values=new_row)
