from flask import Flask, Response, jsonify, request, send_from_directory, url_for
# flask version 2.2.2 cek untuk send_from_directory atau fungsi lainnya
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
from flask_oauthlib.client import OAuth
# from flask_oauthlib.client import OAuth
from flask_caching import Cache
import redis


app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia'
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app, resources={r"/events": {"origins": "*"}})

app.config['CACHE_TYPE'] = 'redis'
app.config['CACHE_REDIS_HOST'] = 'localhost'
app.config['CACHE_REDIS_PORT'] = 6379
app.config['CACHE_REDIS_DB'] = 0
app.config['CACHE_REDIS_URL'] = 'redis://localhost:6379/0'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

cache = Cache(app)

CORS(app, resources={r"/events": {"origins": "*"}})

model = load_model('ecopc_b3.h5')
yolo_model = load_model("./yolo_model.h5")

supabase_client = get_supabase_client()

# class UserProfile(Form):
#     email = StringField('email')
#     password = PasswordField('password')
#     first_name = StringField('fname')
#     last_name = StringField('lname')

#     def _length(min=None, max=None, message=message):
#         if min == None and max == None and min < len() and max > len():
#             raise ValidationError(f"please fill character > {min} and len of characters < {max}")
        


# class Graph(Form):
#     year = StringField('year', [InputRequired()])

#     def validate_year(form, field):
#         data = field.data
#         if not re.search("^\d+$", field.data):
#             raise ValidationError('field must not contain word')

#         if data == year and len(data) != 4:
#             raise ValidationError('Field must equal with 4 characters')

from flask_mail import Mail, Message
from flask_caching import Cache

app.config['MAIL_SERVER']= 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'tachibanahinata2021@gmail.com'
app.config['MAIL_PASSWORD'] = 'raxz inbs buag hylt'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

app.config['CACHE_TYPE'] = 'simple'

mail = Mail(app)

cache = Cache(app)

@app.route('/email/otp', methods=["POST"])
@cross_origin()
def email_otp():
    
    if request.method == "POST":
        try:
            otp = request.json["otp"]
            email = request.json["email"]
            cached_otp = cache.get("your_otp")
            print(f"Received OTP: {otp}, Cached OTP: {cached_otp}, Email: {email}")
                
            if cached_otp is None:
                return jsonify({'error': 'your current otp already out of time, please send again'}), 404
            if str(otp) == str(cached_otp) and email is not None:
                id_role = supabase_client.table('users_roles').select('id_role').eq('email', email).execute()
                roles = id_role.data
                print(roles)
                for val in roles:
                    data = {
                        'email': email,
                        'role': ['admin'] if val['id_role'] == 1 or val['id_role'] == 2 else ['member']
                    }
                token = jwt.encode(data, app.config['SECRET_KEY'], algorithm='HS256')
                return jsonify({'message': 'Login 200 OK', 'token': token}), 200

            return jsonify({'message': 'aneh'}), 400
        except Exception as e:
            return jsonify({'message': 'otp verified error', 'error_msg': str(e)}), 500

    else:
        return jsonify({'message': 'method not allowed'}), 405


@app.route('/login', methods=['POST'])
@cross_origin()
def login():
    if request.method == "POST":
        email=request.json["email"]
        password=request.json["password"]
        try:
            response = supabase_client.table('users').select('email, password').eq('email', email).execute()
            userFound = response.data[0]
            print(userFound)

            # password registered on the app
            if userFound is not None and password == userFound['password']:

                randomIDVerification = random.randint(100000, 999999)
                msg = Message(subject='Ini kode otp-mu', sender="tachibanahinata2021@gmail.com", recipients=[userFound['email']])
                msg.body = f"{randomIDVerification}"
                mail.send(msg)
                cache.set("your_otp", randomIDVerification, timeout=180)
                return jsonify({"message": "please check your email for otp verification", "email": userFound['email']}), 200

            else:
                return jsonify({'message': 'Invalid credentials'}), 401
        except Exception as e:
            return jsonify({'error': 'Internal server error', 'error_msg': str(e)}), 500

# @app.route('/logout', methods=['POST'])
# @cross_origin()
# def user_logout():
#     if request.method == "POST":
#         try:
#             #user logout
#         except Exception as e:
#             return jsonify({'error_msg': str(e)}), 500

BASE_URL = "https://app.midtrans.com/iris"

@app.route('/account/<wallet>', methods=['GET','POST','PATCH'])
@cross_origin()
def create_disbursment(wallet):
    if request.method == 'POST':
        try: 
            name = request.json['name']
            account_id = request.json['account_id']
            base_url = BASE_URL + "/api/v1/beneficiaries"
            payload = {
                "name": "",
                "account": "gopay",
                "bank": wallet,
                "email": "",
                "alias_name": ""
            }

            payload_json_data = json.dumps(payload)

            server_key = os.getenv('server_key')
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': 'Basic ' + base64(str(server_key + ":"))
                # 'X-Idempotency-Key': 
            }

            if server_key is None:
                return jsonify({'error': 'not authorized'}), 401

            response = request.post(url=base_url, data=payload_json_data, headers=headers)
            res = response.json()
            print(res)

            return jsonify({"data": res}), 200
        
        except Exception as err:
            return jsonify({'error': 'internal server error', 'error_msg': str(e)}), 500

    # elif request.method == "GET":
    #     try:
    #     except Exception as err:
    #         return jsonify({'error': 'internal server error', 'error_msg': str(e)}), 500

    # elif request.method == "PATCH":
    #     try:
    #     except Exception as err:
    #         return jsonify({'error': 'internal server error', 'error_msg': str(e)}), 500

    else:
        return jsonify({'error': 'method not allowed'}), 405

@app.route('/payouts', methods=["POST"])
@cross_origin()
def create_payout():
    if request.method == "POST":
        try:
            name = request.json['name']
            account_id = request.json['account_id']
            base_url = BASE_URL + "/api/v1/beneficiaries"
            payload = {
                "name": "",
                "account": "gopay",
                "bank": wallet,
                "email": "",
                "alias_name": ""
            }

            payload_json_data = json.dumps(payload)

            server_key = os.getenv('server_key')
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': 'Basic ' + base64(str(server_key + ":"))
                # 'X-Idempotency-Key': 
            }

            if server_key is None:
                return jsonify({'error': 'not authorized'}), 401

            response = request.post(url=base_url, data=payload_json_data, headers=headers)
            res = response.json()
            print(res)

            return jsonify({"data": res}), 200
        except Exception as err:
            return jsonify({"error_msg": str(err)}), 500
    else:
        return jsonify({"error": "method not allowed"}), 405

#update transaksi

#bug
@app.route('/transaction/<int:id_transaction>', methods=['PUT'])
@cross_origin()
def update_price(id_transaction):
    if request.method == "PUT":
        
        try:
            price = request.json['price']
            # wallet_id = request.json['wallet_id']
            wallet_id = '365210-satu-20240602'
            if wallet_id is None:
                return jsonify({'error': 'please select wallet to do transactions'}), 400

            create_transaction = datetime.datetime.now().date()
            print(create_transaction)
                
            res = supabase_client.table('transactions').update({'created_at': str(create_transaction), 'price_total': int(price)}).eq('id_transaction', id_transaction).execute()
            updated_data_price = res.data[0]
            if updated_data_price is None:
                return jsonify({"error": "id transaction not found"}), 404
            
            ## update saldo dari wallet user atau pakai trigger sql
            responds = supabase_client.table('wallets').select('balance').eq('id_wallet', wallet_id).execute()
            print(responds)
            balance = responds.data[0]['balance']
            updated_price = balance + price
            update_response = supabase_client.table('wallets').update({'balance': updated_price}).eq('id_wallet', wallet_id).execute()

            if update_response is None:
                return jsonify({'error': 'supabase server error'}), 500
            
            return jsonify({"message": f"price of transaction with id {id_transaction} successfully update to {price}", "wallet": f"insert{price} to wallet {wallet_id}"}), 200
        
        except Exception as e:
            return jsonify({'error': 'internal server error', 'error_msg': str(e)}), 500
    else:
        return jsonify({'error': 'method not allowed'}), 405

@app.route('/admins', methods=['GET'])
@cross_origin()
def get_all_admin():
    if request.method == "GET":
        # users = []

        try:
            res = supabase_client.table('users_roles').select('email').eq('id_role', 1).execute()
            # print(res)
            emails = [data['email'] for data in res.data]

            # print(emails)

            users_data = supabase_client.table('users').select('*').in_('email', emails).execute()
            users_info = {}

            print(users_data)
            for data in users_data.data:
                information = {
                    "fname": data['fname'],
                    "lname": data['lname'],
                    "created_at": data['created_at']
                }
                users_info[data['email']] = information

            return jsonify({"message": users_info}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        
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

# IMAGE_DIR = 'D:\\pkb\\foto-botol'
IMAGE_DIR = 'foto-botol'

# kirim foto
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
        # delete cache
        # cache.delete("id_transaction_cache")

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

#init transaction and give id to esp board
@app.route("/api/transaction/init", methods=["POST"])
@cross_origin()
def initialize_id_transaction():
    if request.method == "POST":
        auth_header = request.headers.get("Authorization")
        print(auth_header)

        if not auth_header:
            return jsonify({'message': "Authorization header missing"}), 401

        token = auth_header.split()[1]

        token = decode_token(token)
        if token is False:
            return jsonify({'message': "no token"}), 401
        
        email = token.get("email")

        generateID = math.floor(100000 + random.random() * (99999999 - 100000))
        print(generateID)
        print(email)

        cache.set("id_transaction_cache", generateID)

        return jsonify({'message': 'id transaction successfully cached'}), 200

    else:
        return jsonify({'message': 'Invalid request method'}), 405

# class BoundBox:
#     def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
#         self.xmin = xmin
#         self.ymin = ymin
#         self.xmax = xmax
#         self.ymax = ymax
#         self.objness = objness
#         self.classes = classes
#         self.label = -1
#         self.score = -1

#     def get_label(self):
#         if self.label == -1:
#             self.label = np.argmax(self.classes)

#         return self.label

#     def get_score(self):
#         if self.score == -1:
#             self.score = self.classes[self.get_label()]

#         return self.score

# def _sigmoid(x):
#     return 1. / (1. + np.exp(-x))

# def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
#     grid_h, grid_w = netout.shape[:2]
#     nb_box = 3
#     netout = netout.reshape((grid_h, grid_w, nb_box, -1))
#     nb_class = netout.shape[-1] - 5
#     boxes = []
#     netout[..., :2]  = _sigmoid(netout[..., :2])
#     netout[..., 4:]  = _sigmoid(netout[..., 4:])
#     netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
#     netout[..., 5:] *= netout[..., 5:] > obj_thresh

#     for i in range(grid_h*grid_w):
#         row = i / grid_w
#         col = i % grid_w
#         for b in range(nb_box):
#             # 4th element is objectness score
#             objectness = netout[int(row)][int(col)][b][4]
#             if(objectness.all() <= obj_thresh): continue
#             # first 4 elements are x, y, w, and h
#             x, y, w, h = netout[int(row)][int(col)][b][:4]
#             x = (col + x) / grid_w # center position, unit: image width
#             y = (row + y) / grid_h # center position, unit: image height
#             w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
#             h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
#             # last elements are class probabilities
#             classes = netout[int(row)][col][b][5:]
#             box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
#             boxes.append(box)
#     return boxes

# def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
#     new_w, new_h = net_w, net_h
#     for i in range(len(boxes)):
#         x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
#         y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
#         boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
#         boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
#         boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
#         boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

# def _interval_overlap(interval_a, interval_b):
#     x1, x2 = interval_a
#     x3, x4 = interval_b
#     if x3 < x1:
#         if x4 < x1:
#             return 0
#         else:
#             return min(x2,x4) - x1
#     else:
#         if x2 < x3:
#             return 0
#         else:
#             return min(x2,x4) - x3

# def bbox_iou(box1, box2):
#     intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
#     intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
#     intersect = intersect_w * intersect_h
#     w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
#     w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
#     union = w1*h1 + w2*h2 - intersect
#     return float(intersect) / union

# def do_nms(boxes, nms_thresh):
#     if len(boxes) > 0:
#         nb_class = len(boxes[0].classes)
#     else:
#         return
#     for c in range(nb_class):
#         sorted_indices = np.argsort([-box.classes[c] for box in boxes])
#         for i in range(len(sorted_indices)):
#             index_i = sorted_indices[i]
#             if boxes[index_i].classes[c] == 0: continue
#             for j in range(i+1, len(sorted_indices)):
#                 index_j = sorted_indices[j]
#                 if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
#                     boxes[index_j].classes[c] = 0

from ultralytics import YOLO

def detect_bottle_esp(photo_filename):
    yolo_roboflow_model = YOLO('yolov8n.pt')

    results = yolo_roboflow_model(photo_filename)

    class_names = yolo_roboflow_model.names

    for r in results:
        class_indices = r.boxes.cls
        for idx in class_indices:
            class_name = class_names[int(idx)]
            if class_name == "bottle":
                return True

    return False

@app.route("/uploads", methods=["POST"])
@cross_origin()
def create_transaction():
    if request.method == "POST":
        auth_header = request.headers.get("Authorization")
        print(auth_header)

        if not auth_header:
            return jsonify({'message': "Authorization header missing"}), 401

        token = auth_header.split()[1]

        token = decode_token(token)
        if token is False:
            return jsonify({'message': "no token"}), 401
        
        email = token.get("email")

        # generateID = math.floor(100000 + random.random() * (99999999 - 100000))
        # print(generateID)
        # print(email)
        # print(request.files.getlist('imageFile'))

        try:
            file = request.files('imageFile')
            # if len(files) == 0:
            #     raise ValueError("No files provided")

        except Exception as err:
            print(f"Error: {str(err)}")
            return jsonify({"error": "bad request, request body seems not match"}), 400

        # cache get id transaction
        id_cache = cache.get("id_transaction_cache")

        if id_cache is None:
            return jsonify({"error": "id_transaction not found, please create transaction first"}), 404

        id_str = str(id_cache)
        generateID = int(id_str)

        try:
            new_folder = os.path.join(IMAGE_DIR, str(generateID))
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
                transaction = {
                    'id_transaction': generateID,
                    'email': email
                }
                supabase_client.table('transactions').insert(transaction).execute()
                # cur.execute('''INSERT INTO transaksi (id, email) VALUES (%s, %s)''', (generateID, email))
                # mysql.connection.commit()

            # for file in files:
            old_name_photo = file.filename
            # if not detect_bottle_esp(old_name_photo):
            #     return jsonify({"error: bottle not found"}), 404

            uuid_filename = uuid.uuid4()
            file_path = uuid_filename
                    
            ext_file = old_name_photo.split(".")[1]
            new_name_photo = f"{str(uuid_filename)}.{ext_file}"

            img_tf_data = {
                # 'id_bottle': 2,
                'id_transaction': generateID,
                'file_path': new_name_photo
            }
            res = supabase_client.table('transactions_bottles').insert(img_tf_data).execute()

            try:
                file.save(os.path.join(IMAGE_DIR, str(generateID), old_name_photo))
            except Exception as error_file_saved:
                print(f"error saving file: {error_file_saved}")
                return jsonify({"error": "internal server error, can't save photo"}), 500

            pwd = os.getcwd()
            old_abs_path = f"{pwd}/{IMAGE_DIR}/{generateID}/{old_name_photo}"
            new_abs_path = f"{pwd}/{IMAGE_DIR}/{generateID}/{new_name_photo}"
            os.rename(old_abs_path, new_abs_path)

            return jsonify({'message': 'Sukses mengupload', 'transaction_id': generateID}), 200

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

@app.route("/paid", methods=["GET"])
@cross_origin()
def statistic_controller():
    if request.method == "GET":
        role = get_role_query()
        # print(role)
        un, undo, year = get_query_params()
        cache_key = f"{role}-{year}"
        cache_data = cache.get(cache_key)
        if cache_data:
            cache.delete(cache_key)
            return jsonify({"data": cache_data}), 200

        your_datas = graph_transaction_by_day(role)
        if your_datas is not None and isinstance(your_datas, list):
            cache.set(cache_key, your_datas)
            return jsonify({"data": your_datas}), 200

def graph_transaction_by_day(role):
    try:
        auth_header = request.headers.get("Authorization")
        token = auth_header.split()[1]

        token = decode_token(token)
        if token is False:
            return jsonify({'message': "no token"}), 401
            
        email = token.get("email")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    start_date, end_date, year = get_query_params()

    if start_date is None and end_date is None and year is None:
        return jsonify({"error": "bad request, one parameter needed"}), 400
    try:
        your_datas = []
        dates = []
        for i in range(1, 13):
            start_date = f'{year}-{i:02d}-01'
            if i in {1, 3, 5, 7, 8, 10, 12}:
                end_date = f'{year}-{i:02d}-31'
            elif i == 2:
                end_date = f'{year}-{i:02d}-28'
                # also later add kabisat
            else:
                end_date = f'{year}-{i:02d}-30'
            dates.append([start_date, end_date])
        
        for val in dates:
            sum = 0
            print(f"{val[0]}, {val[1]}")
            
            try: 
                if role != "admin":
                    graph_transaction, count = supabase_client.table('transactions').select("*").eq('email', email).gte("created_at", val[0]).lte("created_at", val[1]).execute()
                else:
                    graph_transaction, count = supabase_client.table('transactions').select("*").gte("created_at", val[0]).lte("created_at", val[1]).execute()
            except Exception as e:
                return jsonify({"error": "transaction not found", "error_msg": str(e)}), 404

            flag = None
            unnecessary, transaction = graph_transaction
            print(transaction)
            for data in transaction:
                if flag is None:
                    flag = data['created_at']
                    flag = int(flag[5:7])  # bug fixed
                sum += data['price_total']
            obj = dict(month=flag, price=sum)
            your_datas.append(obj)
        
        return your_datas
    
    except Exception as e:
        return jsonify({"message": str(e)}), 500

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

#view all verified wallet
@app.route("/wallets", methods=["GET"])
@cross_origin()

def all_wallets_from_user():
    if request.method == "GET":
        try:
            email = get_user_credential(request)
            if email == "no token" or email is None:
                return jsonify({'error': 'user not authorized'}), 401

            res = supabase_client.table('wallets').select('*').execute()
            data = res.data[0]

            if data is None:
                return jsonify({"error": "wallet not found"}), 404
                
            return jsonify({"data": data}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "method not allowed"}), 405

#view detail of verified wallet
@app.route("/wallet/<id_wallet>", methods=["GET"])
@cross_origin()

def view_specific_wallet(id_wallet):
    if request.method == "GET":
        try:
            email = get_user_credential(request)
            # if email is None:
            #     return jsonify({'error': 'internal server error'}), 500

            if email == "no token" or email is None:
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


# oauth = OAuth(app)

# @app.route("/google_auth")
# @cross_origin()
# def login_with_google():
#     google = oauth.remote_app(
#     'google',
#     consumer_key=google_client_id,
#     consumer_secret=google_client_secret,
#     request_token_params={
#         'scope': 'email',
#     },
#     base_url='https://www.googleapis.com/oauth2/v1/',
#     request_token_url=None,
#     access_token_method='POST',
#     access_token_url='https://accounts.google.com/o/oauth2/token',
#     authorize_url='https://accounts.google.com/o/oauth2/auth',
#     )   

#     return google.authorize(callback=url_for('authorized', _external=True))

# @app.route('/google_auth/authorized')
# def authorized():
#     response = google.authorized_response()
#     if response is None or response.get('access_token') is None:
#         return 'Login failed.'

#     # session['google_token'] = (response['access_token'], '')
#     # account = (response['access_token'], '')
#     me = google.get('userinfo')
#     # Here, 'me.data' contains user information.
#     # You can perform registration process using this information if needed.

#     return jsonify({"data": me.data}), 200


# @app.route('/facebook')
# @cross_origin()
# def login_with_facebook():
#     FACEBOOK_CLIENT_ID = os.environ.get('FACEBOOK_CLIENT_ID')
#     FACEBOOK_CLIENT_SECRET = os.environ.get('FACEBOOK_CLIENT_SECRET')
#     oauth.register(
#         name='facebook',
#         client_id=FACEBOOK_CLIENT_ID,
#         client_secret=FACEBOOK_CLIENT_SECRET,
#         access_token_url='https://graph.facebook.com/oauth/access_token',
#         access_token_params=None,
#         authorize_url='https://www.facebook.com/dialog/oauth',
#         authorize_params=None,
#         api_base_url='https://graph.facebook.com/',
#         client_kwargs={'scope': 'email'},
#     )
#     redirect_uri = url_for('facebook_auth', _external=True)
#     return oauth.facebook.authorize_redirect(redirect_uri)

# @app.route('/facebook/auth/')
# def facebook_auth():
#     token = oauth.facebook.authorize_access_token()
#     resp = oauth.facebook.get(
#         'https://graph.facebook.com/me?fields=id,name,email,picture{url}')
#     profile = resp.json()
#     print("Facebook User ", profile)
#     return redirect('/')

#update saldo user atau menggunakan trigger sql


#tes
@app.route("/hello", methods=["GET"])
@cross_origin()
def hello():
    return jsonify({"message": "hello"}), 200
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

        
