# TFRecord read/write

> tfrecord 란 텐서플로우의 학습데이타 등을 저장하기 위한 **바이너리** 데이타포멧 으로, google의 [Protocol Buffer](http://bcho.tistory.com/1182) 포멧으로 데이터를 Serialize하여 저장한다. 또한 이미지를 읽는 과정에서 decoding 과정이 포함됨으로 비 효율적 일 수 있다.

## tfrecord 파일 만들기

tf.train.Example 에 Feature를 딕셔너리로 정의후에 tf.train.Example 객체를 TFRecordWriter로 파일에 저장한다.

```python
tf_example = tf.train.Example(
    features = tf.train.Features(
        feature={
            'image': dataset_util.int64_feature(value),
            'label': dataset_util.bytes_feature(value),
            'name': dataset_util.int64_list_feature(value),
            'name': dataset_util.bytes_list_feature(value),
            'name': dataset_util.float_list_feature(value)
        }
    )
)
```

여기서 
```python
dataset_util.int64_feature(value) 
``` 
를 보면 실제로는 
```python 
tf.train.Feature(int64_list=tf.train.Int64List(value=values))
```
로 구현되어 있다

다음은 이렇게 만든 ```tf.train.Example``` 객체를 ```tf.python_io.TFRecoderWriter``` 를 이용해 다음과 같이 저장시킨다.
```python
writer = tf.python_io.TFRecordWriter(tfrecord_filename)
writer.write(tf_example.SerializeToString())
```

[정말 잘 정리되어 있는 곳](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)