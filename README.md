Пример использования докер файла: docker run -v C:\Projects\Hakaton\docker:/test -v C:\Projects\Hakaton\docker:/results dockerfile
results будет лежать внутри докер образа

Если без докера:
Закидываем в одну директорию с данными файлами папку test с интересующими нас изображениями
и запускаем main.py, автоматически создасться папка /results c results.txt 
