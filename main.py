from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin
from flask_mysqldb import MySQL
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import os
import time
import json
import random
import math
import jwt
import string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia'

app.config['MYSQL_HOST'] = '127.0.0.2'
app.config['MYSQL_PORT'] = 8111
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'eco_plastic_cycle'
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app, resources={r"/events": {"origins": "http://localhost:5173"}})

model = load_model('ecopc_b3.h5')

mysql = MySQL(app)

@app.route('/register', methods=['POST'])
@cross_origin()
def register():
    cur = mysql.connection.cursor()
    if request.method == "POST":
        email = request.json['email']
        password = request.json['password']
        fname = request.json['nama_depan']
        lname = request.json['nama_belakang']
        cur.execute('''INSERT INTO users (email, password, nama_depan, nama_belakang) VALUES (%s, %s, %s, %s)''', (email, password, fname, lname))
        mysql.connection.commit()
        cur.close()
        return jsonify({'message': 'Data added successfully'})

@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    cur = mysql.connection.cursor()
    if request.method == "POST":
        email = request.json['email']
        password = request.json['password']
        cur.execute('''SELECT email, password FROM users WHERE email = %s''', (email,))
        data = cur.fetchone()
        if data and password == data[1]:
            token = jwt.encode({'email': email}, app.config['SECRET_KEY'], algorithm='HS256')
            cur.close()
            return jsonify({'token': token})
        else:
            cur.close()
            return jsonify({'message': 'Invalid credentials'}), 401

IMAGE_DIR = 'D:\\pkb\\foto-botol'

@app.route('/images/<int:transaction_id>/<filename>')
def serve_image(transaction_id, filename):
    directory = os.path.join(IMAGE_DIR, str(transaction_id))
    return send_from_directory(directory, filename)

def prediction(image_file):
    img = Image.open(image_file).convert("RGB").resize((32, 32))
    x = preprocess_input(np.expand_dims(np.array(img), axis=0))
    cleo_test = model.predict(x)
    classes_x = np.argmax(cleo_test, axis=1)
    return [1, 2, 3][classes_x[0]]

@app.route("/events/<int:transaction_id>")
@cross_origin()
def sse(transaction_id):
    def image_from_os(transaction_id):
        image_map = list_images(transaction_id)
        for image_file, image_url in image_map.items():
            predict_val = prediction(os.path.join(IMAGE_DIR, str(transaction_id), image_file))

            with app.app_context():
                cur = mysql.connection.cursor()
                cur.execute('''SELECT * FROM barang WHERE id_barang = %s''', (predict_val,))
                data = cur.fetchone()
                print(data)
                cur.close()

            data = {"image_url": image_url, "prediction": data[1], "price": data[2]}
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(2)
        yield "data: {\"status\": \"END\"}\n\n"

    return Response(image_from_os(transaction_id), mimetype='text/event-stream')

def list_images(transaction_id, filename=None):
    directory = os.path.join(IMAGE_DIR, str(transaction_id))
    files = os.listdir(directory)
    image_files = [file for file in files if os.path.isfile(os.path.join(directory, file))]
    image_urls = [f"http://localhost:5000/images/{transaction_id}/{file}" for file in image_files]
    return dict(zip(image_files, image_urls))

def decode_token(token=None):
    if token is None:
        return False
    decoded_data = jwt.decode(jwt=token, key='rahasia', algorithms=["HS256"])
    return decoded_data

@app.route("/uploads", methods=["POST"])
@cross_origin()
def create_transaction():
    cur = mysql.connection.cursor()
    if request.method == "POST":
        auth_header = request.headers.get("Authorization")
        token = auth_header.split()[1]

        token = decode_token(token)
        if token is False:
            return jsonify({'message': "no token"}), 401
        
        email = token.get("email")

        generateID = math.floor(100000 + random.random() * (99999999 - 100000))
        print(generateID)
        print(email)
        print(request.files.getlist('file'))

        files = request.files.getlist('file')

        try:
            os.makedirs(os.path.join(IMAGE_DIR, str(generateID)))
            cur.execute('''INSERT INTO transaksi (id, email) VALUES (%s, %s)''', (generateID, email))
            mysql.connection.commit()

            for file in files:
                file_path = file.filename
                cur.execute('''INSERT INTO transaksi_barang (id_barang, id_transaksi, file_path) VALUES (%s, %s, %s)''', (2, generateID, file_path))
                mysql.connection.commit()
                file.save(os.path.join(IMAGE_DIR, str(generateID), file_path))

            cur.close()
            return jsonify({'message': 'Sukses mengupload', 'transaction_id': generateID}), 200

        except Exception as e:
            print(f"Error: {e}")
            cur.close()
            return jsonify({'message': 'Error Server'}), 500

    else:
        return jsonify({'message': 'Invalid request method'}), 405


# give role
# @app.route("<:username>/roles/<:role_id>", methods=["PUT"])
# @cross_origin()
# def set_roles(username, role_id):
#     cur = mysql.connection.cursor()
#     cur.execute('UPDATE')

# filter graph by date in user
@app.route("/paid", methods=["GET"])
@cross_origin()

## for example may month 
def graph_transaction_by_day():
    start_date, end_date, year = get_query_params()
    cur = mysql.connection.cursor()

    # months = []

    # print(start_date)
    if start_date == None and end_date == None and year == None:
        return jsonify({"error": "bad request, one parameter needed"}), 400
    try:
        your_datas = []
        dates = []
        for i in range(1, 13):
            start_date = f'{year}-{i:02d}-01'
            if i in {1, 3, 5, 7, 8, 10, 12}:
                end_date = f'{year}-{i:02d}-31' 
            elif i == 2:
                end_date = f'{year}-{i:02d}-29'

                # // kabisat
            else:
                end_date = f'{year}-{i:02d}-30'
            dates.append([start_date, end_date])
        
        for val in dates:
            sum = 0
            print(f"{val[0]}, {val[1]}")
            cur.execute('''SELECT * FROM transaksi WHERE email = %s AND created_at BETWEEN %s AND %s''', ("w@gmail.com", val[0], val[1]))
            rv = cur.fetchall()

            flag = None
            for data in rv:
                if flag is None:
                    flag = data[2]
                    flag = flag.strftime("%m")
                    flag = int(flag[1:])
                sum = sum + data[3]
            obj = dict(month = flag, price = sum)
            your_datas.append(obj)
        
        cur.close()
        # sum = 0
        # for val in rv:
        #     sum = sum + val[3]
        # print(sum)

        return jsonify({"data": your_datas}), 200
    except Exception as e:
        return jsonify({"message": e}), 500

def get_query_params():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    year = request.args.get("year")
    return start_date, end_date, year

# @app.route("/")
# @cross_origin()
# def graph_transaction_by_month():

if __name__ == '__main__':
    app.run(debug=True)
