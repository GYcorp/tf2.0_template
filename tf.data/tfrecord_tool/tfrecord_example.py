import cv2
import tfrecord
import tensorflow as tf

saver = tfrecord.Saver("name_here.tfrecord")

import data

data = data.data()

batch_size = 1
for i, imgs, ptss in data.iterateable(batch_size=batch_size):
    print(i, "Next batch...")

    # 가져온 배치 확인
    key=0
    for i in range( batch_size ):
        print(ptss[i])
        data_dict = {
            'image' : imgs[i],
            'label' : ptss[i]
        }
        saver.save(data_dict)
    # if input('next...') == 'q':
    #     break


reader = tfrecord.Reader("name_here.tfrecord")

reader.shuffle()
reader.batch(1)

iterator = reader.make_one_shot_iterator()
image, label = iterator.get_next()
image, label = tf.Session().run([image, label])
print('image: ', image, 'label: ', label)