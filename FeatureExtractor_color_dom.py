import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import normalize
import warnings

class FeatureExtractor():
   def __init__(self, image_path=str, params=None, use_face_mask=False):
      self.image_path = image_path
      self.image = cv2.imread(image_path)      
      self.use_face_mask = use_face_mask
      self.params = None
      if params is not None:
         {'feature_detector':'', 'desc_vector':''}
         self.params = {}
         try:
            self.params['feature_detector'] = params['feature_detector']
            self.params['desc_vector'] = params['desc_vector']
         except KeyError:
            print('Input parameters is a dictionary consisting of keys \'feature_detector\' \
               and \'desc_vector\'')
   
#
#------- feature selection
#
   def _getFastFeatures(self, image=[]):
      if image == []:
         image = self.image
      image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      fast = cv2.FastFeatureDetector_create()
      fast.setNonmaxSuppression(150)
      if self.use_face_mask:
         face_mask = self.face_detection()[1]
         if face_mask is not None:
            kps = fast.detect(image_bw, face_mask)
         else: 
            kps = fast.detect(image_bw, None)
      else:
         kps = fast.detect(image_bw, None)
      return kps
   
   
   def _getStarFeatures(self, image=[]):
      if image == []:
         image = self.image
      image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      star = cv2.xfeatures2d.StarDetector_create()
      if self.use_face_mask:
         face_mask = self.face_detection()[1]
         if face_mask is not None: 
            kps = star.detect(image_bw, face_mask)
         else: 
            kps = star.detect(image_bw, None)
      else:
         kps = star.detect(image_bw, None)
      return kps
           
   
   def _getOrbFeatures(self, image=[]):
      if image == []:
         image = self.image
      image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      orb = cv2.ORB_create(nfeatures=255)
      if self.use_face_mask:
         face_mask = self.face_detection()[1]
         if face_mask is not None: 
            kps = orb.detect(image_bw, face_mask)
         else: 
            kps = orb.detect(image_bw, None)
      else:
         kps = orb.detect(image_bw, None)
      return kps          

   def _getSkinFeatures(self, image=[], debug=False):
      if image == []:
         image = self.image    
      skin_mask, skinYCrCb = self.skin_detection(image)      
      if debug:
         self.cv_imshow(skinYCrCb, 'Skin ROI') 
         self.cv_imshow(skin_mask, 'Skin mask ROI') 
      skin_feature = self.histogram(skinYCrCb, mask=skin_mask, bins=(8, 12, 3))
      return skin_feature 
      

   def _getSpectrumFeatures(self, image=[], debug=False):
      from sklearn.preprocessing import normalize
      if image == []:
         image = self.image
      image_bw = image if np.shape(image) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      f = np.fft.fft2(image_bw)
      fshift = np.fft.fftshift(f)
      magnitude_spectrum = 20*np.log(np.abs(fshift))
      spectrum_features = np.sort(magnitude_spectrum.flatten())[250::-1]
      spectrum_features = normalize(spectrum_features[:,np.newaxis], axis=0).ravel()      
      if debug:
         plt.subplot(121),plt.imshow(image_bw, cmap = 'gray')
         plt.title('Input Image'), plt.xticks([]), plt.yticks([])
         plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
         plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
         plt.show() 
      return spectrum_features

#   
#------- descriptor vectors
#
   def _orb_desc_vector(self, kps, image=[]):
      if image == []:
         image = self.image
      image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # get model descriptive vector
      orb = cv2.ORB_create(nfeatures=200)
      kps, desc = orb.compute(image_bw, kps)
      return kps, desc
   
   
   def _brief_desc_vector(self, kps, image=[]):
      if image == []:
         image = self.image
      image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # get model descriptive vector
      brief = cv2.DescriptorExtractor_create("BRIEF")
      return brief.compute(image_bw, kps)
         
   def _getFaceHistogram(self, image=[], mask=None, bins=(8, 12, 3)):
      if image == []:
         image = self.image
      img_face = self.face_detection()[0]
      hist = self.histogram(img_face, bins=bins)
      return hist

   def _color_desc_vector(self, image=[]):
      if image == []:
         image = self.image
      image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
      image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      features = []
      bins=(8, 12, 3)
      face_mask, face_coords = self.face_detection()[1:]
      if face_mask is not None:
         (x, y, w, h) = face_coords
         # area boundaries
         H, W = image.shape[:2]
         x_padding = int((W-x)*5/(x))
         y_padding = int((H-y)*5/(y))
         # print('FACE DETECTED!')
         # area 1: face mask
         hist = self.histogram(image, mask=face_mask, bins=bins)
         features.extend(hist)
         # area 2: face padding
         # e_face_mask = np.zeros_like(image_bw)
         # e_face_mask[y-y_padding : y+h+y_padding, x-x_padding : x+w+x_padding] = 1
         # hist = self.histogram(image, mask=e_face_mask-face_mask, bins=bins)
         # features.extend(hist)
         # areas 3-6:
         areas = [(0, x-x_padding, 0, H),
                  (x+w+x_padding, W, 0, H),
                  (x-x_padding, x+w+x_padding, y+h+y_padding, H),
                  (x-x_padding, x+w+x_padding, 0, y-y_padding)]
         for x_start, x_end, y_start, y_end in areas:
            # bound area cordinates
            x_start = 0 if x_start < 0 else x_start
            y_start = 0 if y_start < 0 else y_start
            x_end =  0 if x_end < 0 else x_end
            y_end =  0 if y_end < 0 else y_end
            x_end =  image.shape[1] if x_end > image.shape[1] else x_end
            y_end =  image.shape[0] if y_end > image.shape[0] else y_end
            
            area = image[y_start:y_end, x_start:x_end].copy()
            if area.shape[0] > 0 and area.shape[1]>0:
               color_dom = self.get_dominant_color(area)            
               features.extend(color_dom)
            else:
               color_dom = self.get_dominant_color(image)
               features.extend(color_dom)            
      else:   
         (h, w) = image.shape[:2]      
         (cX, cY) = (int(w * 0.5), int(h * 0.5))
         # divide the image into four rectangles/segments (top-left,
         # top-right, bottom-right, bottom-left)
         segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
         # construct an elliptical mask representing the center of the
         # image
         (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
         ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
         cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 255, 360, -1)
         # loop over the segments
         for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 360, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, mask=cornerMask, bins=bins)
            features.extend(hist)
         # extract a color histogram from the elliptical region and
         # update the feature vector
         hist = self.histogram(image, mask=ellipMask, bins=bins)
         features.extend(hist)
         # return the feature vector
      return np.asarray(features).flatten()
   
#   
#------- feature selection
#   
   def _feature_selector(self, desc_vector, n_features):
      # flatten vector for feature vector
      desc_vector = desc_vector.flatten()
      needed_size = n_features * 64
      if desc_vector.size < needed_size:
         desc_vector = np.concatenate([desc_vector, np.zeros(needed_size - desc_vector.size)])
      else:
         desc_vector = desc_vector[:needed_size]
      return desc_vector
   
   def _keypoint_selector(self, ):
       # keypoint detectors
      if self.params['feature_detector'] == 'fast':
         kps = self._getFastFeatures()
      if self.params['feature_detector'] == 'star':
         kps = self._getStarFeatures()
      if self.params['feature_detector'] == 'orb':   
         kps = self._getOrbFeatures()         
      # Raise warning if no keypoints are captured
      if len(kps)==0:
         warnings.warn('\nKeypoint Warning on image %s: No keypoints detected, \
feature extraction will be applied to the whole image' % self.image_path)
         self.use_face_mask = False
         kps = self._keypoint_selector()
      return kps
   
   
   def feature_extractor(self, n_features=32):
#       print('Feature selection parameters: ', self.params)
      hist_vector = self._color_desc_vector()
      # face_hist_vector = self._getFaceHistogram()
      # skin_features = self._getSkinFeatures()
      # spectrum_vector = self._getSpectrumFeatures()
      desc_vector = []

      if self.params is not None:
         kps = self._keypoint_selector()       
         # Sorting them based on keypoint response value(bigger is better)   
         kps = sorted(kps, key=lambda x: -x.response)[:n_features]   
         # descriptor vectors
         if self.params['desc_vector'] == 'orb':
            kps, desc = self._orb_desc_vector(kps)
         if self.params['desc_vector'] == 'brief':
            kps, desc = self._brief_desc_vector(kps)
            
         # Raise warning if no keypoints are captured
         if desc is None:
            warnings.warn('\nDescriptive Warning on image %s: No descriptive vector generated, \
   feature extraction will be applied to the whole image' % self.image_path)
            kps = self.feature_extractor()
         
         desc_vector = self._feature_selector(desc, n_features)
      features_selected = np.concatenate((hist_vector, 
                                          # face_hist_vector,
                                       #   skin_features, 
                                          # spectrum_vector,
                                          desc_vector))
      return features_selected
           
#      
#------- similarity metrics   
#
   def chi2_distance(self, histA, histB, eps = 1e-10):
      # compute the chi-squared distance
      chi2_dist = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
      return chi2_dist
   
   # helper functions
   def set_params(self, feature_detector, desc_vector):
      self.params['feature_detector'] = feature_detector
      self.params['desc_vector'] = desc_vector

   def histogram(self, image=[], mask=None, bins=[256]):
      if image == []:
         image = self.image
      hist = cv2.calcHist([image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
      hist = hist.flatten()
      # hist = cv2.normalize(hist, hist).flatten()
      return hist   

   def get_dominant_color(self, image, k=4, image_processing_size = None):
      from sklearn.cluster import KMeans
      from collections import Counter
      """
      takes an image as input
      returns the dominant color of the image as a list

      dominant color is found by running k means on the 
      pixels & returning the centroid of the largest cluster

      processing time is sped up by working with a smaller image; 
      this resizing can be done with the image_processing_size param 
      which takes a tuple of image dims as input

      >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
      [56.2423442, 34.0834233, 70.1234123]
      """
      if image == []:
         image = self.image
      #resize image if new dims provided
      if image_processing_size is not None:
         image = cv2.resize(image, image_processing_size, interpolation = cv2.INTER_AREA)

      #reshape the image to be a list of pixels
      image = image.reshape((image.shape[0] * image.shape[1], 3))

      #cluster and assign labels to the pixels 
      clt = KMeans(n_clusters = k)
      labels = clt.fit_predict(image)

      #count labels to find most popular
      label_counts = Counter(labels)

      #subset out most popular centroid
      dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
      return dominant_color

   def face_detection(self, image=[], debug=False):
      if image == []:
         image = self.image
      image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # initialize face detector
      haar_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
      face_cascade = cv2.CascadeClassifier(haar_path)
      faces = face_cascade.detectMultiScale(image_bw, 1.1, 7) 
      if faces is not None:
         try:
            hists = []
            # detect actual face according to skin information
            for f in faces:
               x, y, w, h = f
               face = image[y:y+h, x:x+w]
               # get skin mask
               skin_mask = self.skin_detection(face)[0]
               # calculate 2-bin face mask histogram and keep the second bin count 
               hist = cv2.calcHist([skin_mask],[0],None,[2],[0,256])
               hists.append(hist[1])
            # select the face with the highest count of bin 2 elements
            face_i = np.argmax(hists)
            x, y, w, h = faces[face_i]
            # expand face frame
            face = image[y:y+h, x:x+w]
            # create face mask for discriptive vectors
            face_mask = np.zeros_like(image_bw)
            face_mask[y:y+h, x:x+w] = 1
            if debug:
               self.cv_imshow(face, 'Face ROI') 
            return face, face_mask, (x, y, w, h)
         except:
            warnings.warn('\nWarning on image %s: Face detection failed, \
feature extraction will be applied to the whole image' % self.image_path)
            return None   
      else:
         warnings.warn('\nWarning on image %s: Face not detected, \
feature extraction will be applied to the whole image' % self.image_path)
         return None   
   
   def skin_detection(self, image, debug=False):
      # skin range for YCrCb
      min_YCrCb = np.array([0, 133, 77],np.uint8)
      max_YCrCb = np.array([235, 173, 127],np.uint8)
      image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
      skin_mask = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)
      skinYCrCb = cv2.bitwise_and(image, image, mask = skin_mask)
      if debug:
         self.cv_imshow(skinYCrCb, 'Face ROI') 
         self.cv_imshow(skin_mask, 'Skin mask ROI') 
      return skin_mask, skinYCrCb
      
   def cv_imshow(self, image, title=""):
      if len(np.shape(image))==2:
         plt.title(title)
         plt.imshow(image, cmap='gray')
      if len(np.shape(image))==3 and image.shape[2]==3:
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         plt.title(title)
         plt.imshow(image)
      return plt.show()
      