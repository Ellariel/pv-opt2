from flask import *
#from flask_login import *
#from flask_login import UserMixin
#from flask_sqlalchemy import SQLAlchemy
#from flask_httpauth import HTTPBasicAuth
#from flask.ext.login import LoginManager
from flask import render_template
from werkzeug.utils import secure_filename
#from werkzeug.datastructures import FileStorage
#from flaskwebgui import FlaskUI
#import jwt
import sys
import random
import threading
import pandas as pd
import os
import multiprocessing
import gc
#import time
#from main import *

from utils import get_hash, get_encoded_img, make_timeserie_figure, make_metrics_figure
import main

class CalculationThread(threading.Thread):
    def __init__(self, thread_id, calculation_results):
        self.thread_id = thread_id
        self.calculation_results = calculation_results
        self.calculation_results[self.thread_id] = ''
        self.progress = 0
        self.finished = False
        self.started = False
        super().__init__()

    def run(self):
        self.started = True
        print(f"{self.thread_id} - is started")
        
        # Your exporting stuff goes here ...
        # for _ in range(13):
        #     time.sleep(1) 
        #     self.progress += 10
        #     if self.progress > 100:
        #         self.progress = 0
        
        # main.init_components(base_dir)
        main.calculate(base_dir)
        #time.sleep(3)    
                
        self.progress = 100
        self.calculation_results[self.thread_id] = f"{self.thread_id} - is finished\n" + main.log
        self.finished = True
        print(f"{self.thread_id} - is finished")

base_dir = './'
upload_dir = os.path.join(base_dir, 'uploaded')
#files_types = ['consumption_file', 'production_file', 'excel_file']
files_dir = {'consumption_file': os.path.join(upload_dir, 'consumption'),
             'production_file': os.path.join(upload_dir, 'production'),
             'excel_file': upload_dir}
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(files_dir['consumption_file'], exist_ok=True)
os.makedirs(files_dir['production_file'], exist_ok=True)
os.makedirs(files_dir['excel_file'], exist_ok=True)

figures_dir = os.path.join(base_dir, 'static/figures')
os.makedirs(figures_dir, exist_ok=True)

calculation_threads = {}
calculation_results = {}
#files_types = ['consumption_file', 'production_file', 'building_file',
#               'location_file', 'equipment_file', 'battery_file', 'excel_file']



#os.environ['FLASK_RUN_PORT'] = '8000'
#os.environ['FLASK_RUN_HOST'] = "127.0.0.1"

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__, template_folder='./templates', static_folder='./static')
    
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
app.config['SECRET_KEY'] = '!secret_key!'
app.config['UPLOAD_FOLDER'] = upload_dir
app.config['MAX_CONTENT_PATH'] = 10 * 1000 * 1000
app.config['EXPLAIN_TEMPLATE_LOADING'] = True

main.init_components(base_dir)

@app.route('/')
#@auth.login_required
def index_page(**args):
    args.update({'log': main.log+'\n'})
    return render_template('index.html', **args)

@app.route('/api/v1.0/update')#, methods=['GET', 'POST'])
#@auth.login_required
def update_config():
    try:
        _dict = dict(request.values)
        if len(_dict):
            for k, v in main.config.items():
                if k in _dict:
                    _dict[k] = type(v)(_dict[k])  
            print(f'config update request: {main.config} -> {_dict}')         
            main.config.update(_dict)
        return jsonify({**main.config, 'exception': ''})
    except Exception as e:
        return jsonify({**main.config, 'exception': str(e)})
        
@app.route('/api/v1.0/upload', methods=['GET', 'POST'])
#@auth.login_required
def upload_files():
    uploaded = False
    if request.method == 'POST':
        for ft in files_dir.keys():
            if ft in request.files:
                f = request.files[ft]
                filename = secure_filename(f.filename)
                if len(filename):
                    filename = os.path.join(files_dir[ft], filename)
                    print(f'{f.filename} -> {filename}')
                    f.save(filename) #ft.split('_')[0] + ['.xlsx' if 'excel' in ft else '.csv'][0]))
                    uploaded = True
        if uploaded:
            main.init_components(base_dir, files_dir)
            print('Files uploaded successfully!')
        return index_page(files_uploaded='\nFiles uploaded successfully!\n', log=main.log)+'\n'

@app.route('/api/v1.0/calculate')
#@auth.login_required
def api_calculate():
    global exporting_threads
    for thread_id in list(calculation_threads.keys()):
        if calculation_threads[thread_id].finished:
            del calculation_threads[thread_id]
            print(f"{thread_id} - is removed")
    thread_id = random.randint(0, 10000)
    calculation_threads[thread_id] = CalculationThread(thread_id, calculation_results)
    print(f'Calculations are initialized: {thread_id}.')
    return {'task_id': thread_id, 'exception': ''}

@app.route('/api/v1.0/results/<int:thread_id>')
#@auth.login_required
def api_results(thread_id):
    global calculation_results
    print(f'Results are requested: {thread_id}.')
    if thread_id in calculation_results:
        results = calculation_results[thread_id]
        return jsonify({'task_id': thread_id, 'results': results, 'exception': ''})
    else:
        return jsonify({'task_id': thread_id, 'results': '', 'exception': 'No such results!'})
    
@app.route('/api/v1.0/progress/<int:thread_id>')
#@auth.login_required
def api_progress(thread_id):
    global calculation_threads
    
    #collected = gc.collect()
    #if collected:
    #        print("Garbage collector: collected",
    #      "%d objects." % collected)     
    
    
    if thread_id in calculation_threads:
        thread = calculation_threads[thread_id]
        if not thread.is_alive() and not thread.started:
            thread.start()
            print(f'Calculations are started: {thread_id}.')
        return jsonify({'task_id': thread_id, 'progress': thread.progress, 'finished': thread.finished, 'exception': ''})
    else:
        return jsonify({'task_id': thread_id, 'exception': 'Wrong task_id!'})

@app.route('/api/v1.0/table/<string:table_name>')
#@auth.login_required
def api_table(table_name):
    if table_name in main.data_tables:
        table = main.data_tables[table_name]
        cols = [{'title': i} for i in table.columns]
        print(f'Table is requested: {table_name}.')
        if len(table):
            data = [v.fillna('-').values.tolist() for k, v in table.iterrows()]
            #print(data)
            return jsonify({'table_name': table_name, 'data': data, 'cols': cols, 'exception': ''})
        else:
            return jsonify({'table_name': table_name, 'data': '', 'cols': cols, 'exception': 'Table is empty!'})
    else:
        return jsonify({'table_name': table_name, 'data': '', 'cols': '', 'exception': 'No such a table name!'})

@app.route('/api/v1.0/figure/<string:source>/<string:data_type>/<string:uuid>')
#@auth.login_required
def api_figure(source, data_type, uuid):
    #print(source, data_type, uuid)
    try:
        d = pd.Series()
        if source == 'building':
            d = main.buildings[uuid][data_type]
        elif source == 'solution':
            _, s = main.load_solutions(main.data_tables['solution_data'], uuid, storage=main.solution_dir)
            if s:
                if data_type == 'metrics':
                    d = pd.DataFrame([i['metrics'] for i in s])
                else:
                    d = s[0]['building'][data_type]
        if len(d):
            file_name = os.path.join(figures_dir, get_hash(d)+'.png')
            if not os.path.exists(file_name):
                if data_type == 'metrics':
                    make_metrics_figure(d, file_name) 
                else:
                    make_timeserie_figure(d, file_name)
            return jsonify({'image_url': get_encoded_img(file_name), 'exception': ''})
        return jsonify({'image_url': '', 'exception': 'Image error: no matched uuid!'})
    except Exception as e:
        jsonify({'image_url': '', 'exception': f'Image error: {str(e)}'})

#########################################################################

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app.run(host='127.0.0.1', port=5003, debug=True)
    
    

'''
        
        <script>    
const canvasMask = document.getElementById('myCanvas');
const ctxcanvasMask = canvasMask.getContext('2d');

document.getElementById("getMask").addEventListener('click', () => {

    fetch('http://127.0.0.1:5000/getImage')
    .then(res => res.json())
    .then(data => {
        var img = new Image();
        img.src = 'data:image/jpeg;base64,' + data.image_url;
        img.onload = () => ctxcanvasMask.drawImage(img, 0, 0);
    })
    .catch(err => alert("PROBLEM\\n\\n" + err));
    
});
</script>







login_manager = LoginManager()
login_manager.init_app(app)

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(300), nullable=False)

    def verify_password(self, password):
        return self.password == password

    def generate_auth_token(self, expires_in = 600):
        return jwt.encode(
            {'id': self.id, 'exp': time.time() + expires_in},
            app.config['SECRET_KEY'], algorithm = 'HS256')

    @staticmethod
    def verify_auth_token(token):
        try:
            data = jwt.decode(token,
                              app.config['SECRET_KEY'],
                              algorithms=['HS256'])
        except:
            return
        return User.query.get(data['id'])

db.init_app(app)
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    user = None
    token = request.headers.get('Authorization')
    if token:
        user = User.verify_auth_token(token)
    if not user:
        user = User.query.filter_by(username = username).first()
        if not user or not user.verify_password(password):
            return False
    g.user = user
    return True

@app.route('/api/v1.0/register/', methods=['POST'])
def register():
    try:
        args = request.get_json()
        name = args.get('name')
        pwd = args.get('password')
        user = User.query.filter_by(username=name).first()
        if user:
            return jsonify('Username already registered'), 201
        user = User(username = name, password = pwd)
        db.session.add(user)
        db.session.commit()
        return jsonify({'name': user.username }), 201
    except Exception as e:
        return jsonify({'exception': str(e)}), 400
        
@app.route('/api/v1.0/login/', methods=['POST'])
def login():
    try:
        args = request.get_json()
        name = args['name']
        pwd = args['password']
        user = User.query.filter_by(username=name).first()
        if not user:
            return jsonify('User is not registered'), 400
        if user.password == pwd:
            login_user(user)
            return jsonify({'name': user.username }), 200
        else:
            return jsonify('Password is incorrect'), 400
    except Exception as e:
        return jsonify({'exception': str(e)}), 400

@app.route('/api/v1.0/get_token/')
@auth.login_required
def get_token():
    try: 
        token = g.user.generate_auth_token()
        return jsonify({'token': token})
    except Exception as e:
        return jsonify({'exception': str(e)}), 400

@app.route('/api/v1.0/logout/', methods=['POST'])
@auth.login_required
def logout():
    try: 
        logout_user()
        return jsonify('Logged out'), 200
    except Exception as e:
        return jsonify({'exception': str(e)}), 400
'''
pass