import keras.backend as K
from keras import callbacks
import numpy as np
import cv2
import matplotlib.pyplot as plt



class vis_layer(callbacks.Callback):
	def __init__(self, imgs, labels, layers, results_path, batch_size=6):
		super(vis_layer, self).__init__()
		self.X = imgs
		self.Y = labels
		self.batch_size = batch_size
		self.layers = layers
		self.output_path = results_path
	def on_epoch_end(self, epoch, logs=None):

		# random_idx = np.random.choice(np.arange(0,len(self.X)), self.batch_size)
		random_idx = [75,  86, 22, 268, 180, 210]
		
		img_batch = []
		for i in random_idx:
			img_batch.append(self.X[i])
		img_batch = np.array(img_batch)
		img_batch_row = np.concatenate(img_batch, axis = 1)

		pred = self.model.predict(img_batch)
		print('Predicted')
		
		l_rows = []
		step = -1
		for l in self.layers:
			l_model = K.function([self.model.layers[0].input], [self.model.layers[l].output])
			l_out = l_model([img_batch])[0]
			l_out_img = []
			step += 1
			
			for i in range(self.batch_size):
				out_mean = np.average(l_out[i], axis= 2)
				out_norm = cv2.normalize(out_mean ,0 , 255, norm_type=cv2.NORM_MINMAX)
				l_out_img.append(out_norm)
			l_out_img = np.concatenate(l_out_img, axis = 1)
			l_rows.append(l_out_img)

		num_rows = len(l_rows) + 1


		plt.subplot(num_rows, 1, 1)
		plt.imshow(img_batch_row)
		plt.title('Intermediate layer visualization')
		plt.xticks([])
		plt.yticks([])
		plt.ylabel('Original')

		for i in range(len(l_rows)):
			plt.subplot(num_rows, 1, i+2)
			plt.imshow(l_rows[i])
			plt.xticks([])
			plt.yticks([])
			plt.ylabel('layer {}'.format(self.layers[i]))

		plt.savefig(self.output_path +'/Layer_vis_epoch_{}.png'.format(epoch))


