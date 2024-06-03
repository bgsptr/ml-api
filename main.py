from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin
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
import datetime

from config.spbs import get_supabase_client

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia'
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app, resources={r"/events": {"origins": "http://localhost:5173"}})

model = load_model('ecopc_b3.h5')

supabase_client = get_supabase_client()

@app.route('/transaction/<int:id_transaction>', methods=['PUT'])
@cross_origin()
def update_price(id_transaction):
    if request.method == "PUT":
        
        try:
            price = request.json['price']

            res = supabase_client.table('transactions').update({'price': price}).eq('id_transaction', id_transaction).execute()
            updated_data_price = res.data[0]
            if updated_data_price is None:
                return jsonify({"error": "id transaction not found"}), 404
            
            ## update saldo dari wallet user atau pakai trigger sql
            
            return jsonify({"message": f"price of transaction with id {id_transaction} successfully update to {price}"}), 200
        
        except Exception as e:
            return jsonify({'error': 'Internal server error', 'error_msg': str(e)}), 500
        
@app.route('/register', methods=['POST'])
@cross_origin()
def register():
    if request.method == "POST":
        try:
            email = request.json['email']
            password = request.json['password']
            fname = request.json['fname']
            lname = request.json['lname']

            user_data = {
                'email': email,
                'password': password,
                'fname': fname,
                'lname': lname
            }
            res = supabase_client.table('users').insert(user_data).execute()
            return jsonify({'message': f"Data with email {res.data[0]['email']} successfully"})
        
        except Exception as e:
            return jsonify({'error': 'Internal server error', 'error_msg': str(e)}), 500
        
@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    if request.method == "POST":
        try:
            email = request.json['email']
            password = request.json['password']
            response = supabase_client.table('users').select('email, password').eq('email', email).execute()
            userFound = response.data[0]
            print(userFound)
            if userFound and password == userFound['password']:
                token = jwt.encode({'email': email}, app.config['SECRET_KEY'], algorithm='HS256')
                return jsonify({'token': token})
            else:
                return jsonify({'message': 'Invalid credentials'}), 401
        except Exception as e:
            return jsonify({'error': 'Internal server error', 'error_msg': str(e)}), 500

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
                # cur = mysql.connection.cursor()
                # cur.execute('''SELECT * FROM barang WHERE id_barang = %s''', (predict_val,))
                # data = cur.fetchone()
                # print(data)
                # cur.close()

                res = supabase_client.table('bottles').select('*').execute()
                data = res.data[0]

            data = {"image_url": image_url, "prediction": data['bottle_name'], "price": data['price']}
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
            transaction = {
                'id': generateID,
                'email': email
            }
            supabase_client.table('transactions').insert(transaction).execute()
            # cur.execute('''INSERT INTO transaksi (id, email) VALUES (%s, %s)''', (generateID, email))
            # mysql.connection.commit()

            for file in files:
                file_path = file.filename
                img_tf_data = {
                    'id_bottle': 2,
                    'id_transaction': generateID,
                    'file_path': file_path
                }
                res = supabase_client.table('transactions_bottles').insert(img_tf_data).execute()
                file.save(os.path.join(IMAGE_DIR, str(generateID), file_path))

            return jsonify({'message': 'Sukses mengupload', 'transaction_id': res.data[0]['id_transaction']}), 200

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'message': 'Error Server'}), 500

    else:
        return jsonify({'message': 'Invalid request method'}), 405


# give role
@app.route("/roles/<int:role_id>/<email>", methods=["PUT"])
@cross_origin()
def set_roles(role_id, email):
    # cur = mysql.connection.cursor()
    # cur.execute('''INSERT INTO users_role (id_role, email) VALUES (%s, %s)''', (role_id, email))
    # cur.close()
    role_to_user = {
        'id_role': role_id,
        'email': email
    }
    supabase_client.table('users_role').insert(role_to_user).execute()

    return jsonify({'message': 'ok update role user'}), 200

# filter graph by date in user
@app.route("/paid", methods=["GET"])
@cross_origin()
def statistic_controller():
    role = get_role_query()
    print(role)
    your_datas = graph_transaction_by_day(role)
    if your_datas is not None:
        return jsonify({"data": your_datas}), 200

## for example may month 
def graph_transaction_by_day(role):
    try:
        auth_header = request.headers.get("Authorization")
        token = auth_header.split()[1]

        token = decode_token(token)
        if token is False:
            return jsonify({'message': "no token"}), 401
            
        email = token.get("email")
    except Exception as e:
        return jsonify({"error": e}), 500


    start_date, end_date, year = get_query_params()

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

                # also later add kabisat
            else:
                end_date = f'{year}-{i:02d}-30'
            dates.append([start_date, end_date])
        
        for val in dates:
            sum = 0
            print(f"{val[0]}, {val[1]}")
            
            if role != "admin":
                graph_transaction = supabase_client.table('transaction').select("*").eq('email', email).gte("created_at", val[0]).lte("created_at", val[1]).execute()
                # cur.execute('''SELECT * FROM transaksi WHERE email = %s AND created_at BETWEEN %s AND %s''', (email, val[0], val[1]))
            else:
                graph_transaction = supabase_client.table('transaction').select("*").gte("created_at", val[0]).lte("created_at", val[1]).execute()
                # cur.execute('''SELECT * FROM transaksi created_at BETWEEN %s AND %s''', (val[0], val[1]))
            # rv = cur.fetchall()

            flag = None
            for data in graph_transaction.data[0]:
                if flag is None:
                    flag = data["created_at"]
                    flag = flag.strftime("%m")
                    flag = int(flag[1:])
                sum = sum + data[3]
            obj = dict(month = flag, price = sum)
            your_datas.append(obj)
        
        return your_datas
    
    except Exception as e:
        return jsonify({"message": e}), 500

def get_query_params():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    year = request.args.get("year")
    return start_date, end_date, year

def get_role_query():
    role = request.args.get("role")
    return role

# @app.route("/")
# @cross_origin()
# def graph_transaction_by_month():

@app.route("/profile/<email>", methods=["PUT", "DELETE"])
@cross_origin()
def edit_del_profile(email):
    if request.method == "PUT":

        try:
            # password = request.json['password']
            fname = request.json['fname']
            lname = request.json['lname']

            #select check kalo data sama kirim response aja 200
            message = supabase_client.table('users').select("*").eq('email', email).execute()
            data = message.data[0]
            if data is None:
                return jsonify({'error': 'user with email not found'}), 404
            
            # kalo update to data yang sama langsung return response aja
            if data['fname'] == fname and data['lname'] == lname:
                return jsonify({'message': 'data not need any change'}), 200
            
            ## improve lagi
            update_data = supabase_client.table('users').update({'fname': fname, 'lname': lname}).eq('email', email).execute()
            if update_data.data[0] is None:
                return jsonify({'error': 'internal server error, cant update users'}), 500
                
            # cur.execute('''INSERT INTO users (email, password, nama_depan, nama_belakang) VALUES (%s, %s, %s, %s)''', (email, password, fname, lname))
            # supabase_client.execute('''INSERT INTO users (email, password, nama_depan, nama_belakang) VALUES (%s, %s, %s, %s)''', (email, password, fname, lname))
            # mysql.connection.commit()
            # cur.close()
            return jsonify({'message': 'Data update successfully'}), 200
        except Exception as e:
            return jsonify({'error': 'internal server error', 'error_msg': str(e)}), 500
        
    elif request.method == "DELETE":
        try: 
            res, count = supabase_client.table('users').delete().eq('email', email).execute()
            return jsonify({'message': f"Data with email {email} delete successfully"}), 200
        except Exception as e:
            return jsonify({'error': 'internal server error', 'error_msg': str(e)}), 500
    else:
        return jsonify({"error": "api method bad request"}), 405
    
# view all customer, and view all admin(important)
@app.route('/wallet', methods=["POST"])
@cross_origin()
def create_wallet():
    if request.method == "POST":
        try:
            email = get_user_credential(request)
            if email is None:
                return jsonify({'error': 'internal server error'}), 500
            
            if email == "no token":
                return jsonify({'error': 'user not authorized'}), 401
            
            id_wallet = generate_user_id_wallet(email)

            new_wallet = {
                'id_wallet': id_wallet,
                'balance': 0,
                'email': email,
                'status': 'pending'
            }

            wallet_created_response = supabase_client.table('wallets').insert(new_wallet).execute()
            wallet = wallet_created_response.data[0]
            if wallet is not None:
                return jsonify({'message': f"your wallet with id {wallet['id_wallet']} successfully created", 'status': 'OK'}), 200
            
            return jsonify({'error': "internal supabase server error"}), 500
        
        except Exception as e:
            return jsonify({'error': 'internal server error', 'error_msg': str(e)}), 500
        
    else:
        return jsonify({'error': 'status bad request, method not allowed'}), 405

def generate_user_id_wallet(email):
    username = email.split("@")

    fetch_wallet = str(datetime.datetime.now())
    fetch_wallet_split = fetch_wallet.split(" ")
    wallet_created_date = fetch_wallet_split[0].replace("-", "")

    #6 digit random wallet id
    generated_wallet_id = random_with_N_digits(6)

    generated_id_str = "{}-{}-{}"
    wallet_id = generated_id_str.format(generated_wallet_id, username[0], wallet_created_date)
    return wallet_id

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return random.randint(range_start, range_end)

def get_user_credential(request):
    try:
        auth_header = request.headers.get("Authorization")
        token = auth_header.split()[1]

        token = decode_token(token)
        if token is False:
            # return jsonify({'message': "no token provided"}), 401
            return f"no token"
            
        email = token.get("email")
        return email
    except Exception as error:
        print(str(error))
        return None
    
@app.route('/wallet/verify/<id_wallet>', methods=["PUT"])
@cross_origin()
def admin_verified_wallet(id_wallet):
    if request.method == "PUT":

        try:
            data = get_user_credential(request)
            if data is None:
                return jsonify({'error': 'internal server error'}), 500
                
            if data == "no token":
                return jsonify({'error': 'user not authorized'}), 401
            
            ## check response role biasanya return beberapa row role karena 1 user has many roles
            role = supabase_client.table("users_roles").select("id_role").eq("email", data).execute()
            # print(role)
            if role is None:
                return jsonify({'error': 'supabase server error'}), 500
            
            if role.data[0]['id_role'] != 1 and role.data[1]['id_role'] != 2:
                return jsonify({'error': 'user is not admin', 'error_msg': 'forbidden action'}), 403
            
            # try catch error supabase update data atau cek role
            updated_status_wallet = {'status': 'verified', 'verified_by_admin': data}
            updated_status = supabase_client.table("wallets").update(updated_status_wallet).eq('id_wallet', id_wallet).execute()
            if updated_status is None:
                return jsonify({'error': "supabase server error"}), 500
            
            return jsonify({'message': 'wallet verified'}), 200
        except Exception as e:
            return jsonify({'error': e.message}), 500
        
    
    else:
        return jsonify({'error': 'method not allowed'}), 405
    
#view saldo dan informasi user
# @app.route("/me/wallet")
# @cross_origin

@app.route("/wallet/<id_wallet>", methods=["GET"])
@cross_origin()

def view_specific_wallet(id_wallet):
    if request.method == "GET":
        try:
            email = get_user_credential(request)
            if email is None:
                return jsonify({'error': 'internal server error'}), 500

            if email == "no token":
                return jsonify({'error': 'user not authorized'}), 401
            
            result = supabase_client.rpc('get_user_specific_wallet_information', {'email_user': email, 'wallet_id_param': id_wallet}).execute()

            print(result)
            data = result.data[0]

            if not data:
                return jsonify({'error': 'wallet not found'}), 404

            return jsonify({'message': data}), 200
        except Exception as e:
            return jsonify({'error': 'internal server error', 'error_msg': str(e)}), 500
    else:
        return jsonify({'error': 'method not allowed'}), 405

#update saldo user atau menggunakan trigger sql


#tes
@app.route("/hello", methods=["GET"])
@cross_origin()
def hello():
    return jsonify({"message": "hello"}), 200
    
if __name__ == '__main__':
    app.run(debug=True)
