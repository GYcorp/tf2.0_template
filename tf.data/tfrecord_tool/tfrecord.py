# https://www.tensorflow.org/guide/datasets#preprocessing_data_with_datasetmap
# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

import tensorflow as tf
import os

'''
tfrecord is XML
but smaller, faster, and simpler.

example
features {
	feature {
		key: "Image"
		value {
			bytes_list {
				value: "\000"
			}
		}
	}
	feature {
		key: "Image_height"
		value {
			int64_list {
				value: 1080
			}
		}
	}
	feature {
		key: "Image_width"
		value {
			int64_list {
				value: 1920
			}
		}
	}
	feature {
		key: "Label"
		value {
			int64_list {
				value: 1
			}
		}
	}
}
'''

class Saver:
	def __init__(self, tfrecords_filename, help=True):
		self.writer = tf.io.TFRecordWriter(tfrecords_filename)

	def _bytes_feature(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	def _float_feature(self, value):
		return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
	def _int64_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def _bytes_list_feature(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
	def _float_list_feature(self, value):
		return tf.train.Feature(float_list=tf.train.FloatList(value=value))
	def _int64_list_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

	def save(self, data_dict):
		if type(data_dict) == dict:
			feature_dict = {}
			for index in data_dict:
				data = data_dict[index]
				data_type = type(data)
				if	 data_type == bytes: feature_dict[index] = self._bytes_feature(data)
				elif data_type == float: feature_dict[index] = self._float_feature(data)
				elif data_type == int  : feature_dict[index] = self._int64_feature(data)
				elif data_type == list :
					list_type = type(data[0])
					if	 list_type == bytes: feature_dict[index] = self._bytes_list_feature(data)
					elif list_type == float: feature_dict[index] = self._float_list_feature(data)
					elif list_type == int  : feature_dict[index] = self._int64_list_feature(data)
					else: raise TypeError("\nNot Support Type : "+str(data_type)+str(list_type)+"\nSupprot types = [bytes, float, int64]")
				else: raise TypeError("\nNot Support Type : "+str(data_type)+"\nSupprot types = [bytes, float, int64]")

			tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
			self.writer.write(tf_example.SerializeToString())
		else:
			raise TypeError("\n it must be 'dict' type")
	
	def exit(self):
		self.writer.close()

class Reader:
	def __init__(self, tfrecord_filenames, feature):	

		self.dataset = tf.data.TFRecordDataset(tfrecord_filenames)
		self.feature = feature

		if type(feature) == dict:
			keys_to_features = {}
			for index in feature:
				data = feature[index]
				data_type = type(data)
				if	 data_type == bytes: keys_to_features[index] = tf.io.FixedLenFeature([], tf.string, default_value="") 
				elif data_type == float: keys_to_features[index] = tf.io.FixedLenFeature([], tf.float32, default_value=0.)
				elif data_type == int  : keys_to_features[index] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
				elif data_type == list :
					list_type = type(data[0])
					if	 list_type == bytes: keys_to_features[index] = tf.io.FixedLenSequenceFeature([], tf.string, default_value=[""], allow_missing=True)
					elif list_type == float: keys_to_features[index] = tf.io.FixedLenSequenceFeature([], tf.float32, default_value=[0.], allow_missing=True)
					elif list_type == int  : keys_to_features[index] = tf.io.FixedLenSequenceFeature([], tf.int64, default_value=[0], allow_missing=True)
					else: raise TypeError("\nNot Support Type : "+str(data_type)+str(list_type)+"\nSupport types = [bytes], [float], [int64]")
				else: raise TypeError("\nNot Support Type : "+str(data_type)+"\nSupport types = bytes, float, int64")
			self.keys_to_features = keys_to_features
		else:
			raise TypeError("\n it must be 'dict' type")

		self.dataset = self.dataset.map(self._parser)

	def _parser(self, record):
		parsed = tf.io.parse_single_example(record, self.keys_to_features)

		parsed_ordered = []
		for index in self.feature:
			parsed_ordered.append(parsed[index])

		return parsed_ordered