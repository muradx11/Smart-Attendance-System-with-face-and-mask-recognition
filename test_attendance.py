from attendance import mark_attendance, get_today_attendance


success, message = mark_attendance("2400101460")

print(message)

records = get_today_attendance()

for record in records:
    print(dict(record))