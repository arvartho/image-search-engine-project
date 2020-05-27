from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from flask import session as sess
from  FeatureExtractor import FeatureExtractor
import numpy as np
import os
import warnings

UPLOAD_FOLDER = './static/images/UPLOADS'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
AVAILABLE_METRICS = {
                    'd_1':'euclidean',  # sqrt(sum((x - y)^2))
                    'd_2':'manhattan',  # sum(|x - y|)
                    'd_3':'hamming',    # N_unequal(x, y) / N_tot
                    'd_4':'chebyshev',  # max(|x - y|)
                    'd_5':'canberra'  # sum(|x - y| / (|x| + |y|))
                  }

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

@app.route('/')
def hello():
    # create start environment
    # return "Hello World!"
    return render_template('index.html')

@app.route('/index')
def show_index():
    return render_template('index.html')

@app.route('/queryImage', methods = ['GET', 'POST'])
def queryImage():
    if request.method == 'POST':
      if 'file' not in request.files:
        flash('No file part')

      file = request.files['filename']
      selection = request.form.get('distance_selection')
      k = int(request.form.get('k_selection'))
      filename = file.filename
      print('Form selection:,',filename, selection)
      if filename and selection and allowed_file(filename):
        s_filename = secure_filename(filename)
        img_filepath = os.path.join(app.config['UPLOAD_FOLDER'], s_filename)
        file.save(img_filepath)
        # get input image features, 'fast' features will be selected by default
        input_img_features = get_input_file_features(img_filepath)
        # get all records from posgres
        query_res = get_all_records()
        descriptor_vectors = []
        for id, filename, filepath, d_vector in query_res:
          descriptor_vectors.append(np.frombuffer(d_vector, dtype=np.dtype('float64')))
        descriptor_vectors = np.asarray(descriptor_vectors)
        # get K neighbors
        print('get_kneighbors', descriptor_vectors.shape, input_img_features.shape)
        distance, neighbors = get_kneighbors(k, selection, descriptor_vectors, input_img_features)
        print('distance', distance, 'neighbors', neighbors)
        # prepare rendered results
        results = [(d, query_res[i][2], query_res[i][1])  for d,i in zip(distance, neighbors)]
        print('Results', results)
        return render_template('results.html', orig_img = img_filepath,  results = results)  
      else:   
        flash('Please select an image and a distance metric')
        return redirect(url_for('http://localhost:5000/index'))
        # return render_template('index.html', message = "Please select an image and a distance metric")  

#
# ------ helper functions
#
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_input_file_features(img_filepath, feature_detector='orb'):
  if feature_detector is not 'fast' or feature_detector is not 'orb':
    warnings.warn('\nFeature detector Warning: No feature detector selected, \
\'fast\' feature detection will be applied')
  params = {'feature_detector':feature_detector, 'desc_vector':'orb'}  
  img_features = FeatureExtractor(img_filepath)
  sample_descr_vector = img_features.feature_extractor()
  return sample_descr_vector

def get_kneighbors(k, metric, X, sample):
  from sklearn.neighbors import NearestNeighbors   
  nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=AVAILABLE_METRICS[metric])

  nbrs.fit(X)
  d, n = nbrs.kneighbors(sample.reshape(1, -1))
  return np.around(d[0], 2), n[0]

def chi2_distance(histA, histB, eps = 1e-10):
   # compute the chi-squared distance
   chi2_dist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
         for (a, b) in zip(histA, histB)])
   return chi2_dist

def get_all_records():
  import psycopg2
  from datetime import datetime

  conn = psycopg2.connect(database="image_db", 
                        user = "postgres", 
                        password = "admin", 
                        host = "localhost", 
                        port = "5432")

  print("Opened database successfully")
  #Creating a cursor object using the cursor() method
  cur = conn.cursor()
  # Setup query
  sql = '''SELECT * from images'''
  start = datetime.now()
  print('Start retrieving query')
  #Executing the query
  cur.execute(sql)
  #Fetching 1st row from the table
  query_res = cur.fetchall()
  print('DB transaction finished in', datetime.now() - start)
  conn.close()
  return query_res

def rendered_results(neighbors, request):
  return [request[i] for i in neighbors]

if __name__=="__main__":
  app.config['SESSION_TYPE'] = 'filesystem'
  sess.init_app(app)
  app.run()