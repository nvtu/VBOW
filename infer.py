from extract_features import gen_sift_features, to_gray
from quantization import generate_term_freq
import numpy as np
from vbow_vectorize import vbow_create
from pathlib import Path
from nearest_neighbor import load_model, nearest_neighs


kmeans_storage = np.load(str(Path.cwd() / 'kmeans_cluster.storage'))
idf = np.load(str(Path.cwd() / 'idf.npy'))
model = load_model()
image_order_filepath = str(Path.cwd() / 'image_path_order.txt')
image_order = [line.rstrip() for line in open(image_order_filepath, 'r').readlines()]

image_path = ''
image = cv2.imread(image_path)
gray_image = to_gray(image)
_, desc = gen_sift_features(gray_image)
tf_feat = generate_term_freq(desc, kmeans_storage)
vbow_feat = vbow_create(tf_feat, idf)
rank_list, distance = nearest_neighs(vbow_feat)

