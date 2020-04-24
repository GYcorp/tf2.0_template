# tf.data 사용하기

> tf.data는 데이터 전처리과정을 최소화해 주는 녀석이다. 하지만 개념이 이해안되 한번 접었던 API다. (없어도 학습 잘만되던데..)

데이터를 불러오는 과정은 다음과 같다.
1. 데이터가 있는 경로를 찾는다.
2. 데이터를 불러오고 사용할수 있는 데이터로 파싱한다.
3. 데이터 전처리를 한다.
4. 어떻게든 불러쓸수있도록 원하는방식으로 데이터를 주는 함수를 만든다.

그리고 학습할때 함수를 잘 불러 써주면 된다.

## 1. 데이터 준비

tfrecord 인 경우
```python
tf.data.TFRecordDataset("./*.tfrecord",...)
```

array 인 경우
```python
tf.data.from_tensor_slices(list)
# 이렇게 하면 list 가 잘 들어가 있는다. string 도 가능하고(경로) 상수도 가능하다. 튜플도 넣을수 있다.
```

## 2. 데이터 전처리

 tfrecord의 경우 바로 입력으로 사용할 수 있지만, image를 읽거나 one_hot 포멧으로 변환해 줘야 하는 경우 reader(읽기. 데이터 생성시) 및 preprocess() 함수를 정의할 수 있다.

```python
# 이미지와 라벨을 반환하는 함수
def _read_py_function(path, label):
    image = read_image(path)
    label = np.array(ladel, dtype=np.uint8)
    return image.astype(np.int32), label

# 이미지의 크기를 변환하는 함수
def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded, [28,28])
    return image_resized, label

```
두개 모두 image 와 label을 입력과 출력으로 받고있다.
왜냐하면 나중에 map 함수로 적용할 것이기 때문이다.

```python
dataset = dataset.map(lambda data_list, label_list: tuple(tf.py_func(_read_py_function, [data_list, label_list], [tf.int32, tf.uint8])))
```
dataset.map(함수) 람다로 하면 호출될때 실행되나보다.

## 3. 데이터 옵션 주기
데이터를 사용하기전에 데이터의 순서를 바꾸거나 shuffle, batch, epoch 등을 정할 수 있다.
```python
# 반복설정하기
dataset = dataset.repeat()
# 순서 섞기 buffer_size 는 섞을 데이터 수 이다. (그냥 다 섞어줄것이지 쯧쯧..)
dataset = dataset.shuffle(buffer_size=( int(len(data_list)*0.4)+3*batch_size ))
```

## 4. batch size 정의하기
손쉽에 원하는 배치사이즈를 만들 수 있다.
```python
dataset = dataset.batch(batch_size)
```

## 5. iterator 정의
iterator란 next() 함수를 호출해서 다음 배치를 얻을 수 있는 편리한 구조이다.
```python
# iterator를 만들었다.
iterator = dataset.make_initializeable_iterator()
# 매번 이 코드를 실행하면 다음 배치의 데이터가 들어올 것 이다.
image_stacked, label_stacked = iterator.get_next()
```

## 6. image 읽기
session 열고 초기화해주고 run 해주면 바로 이미지가 나오는것을 확인할 수 있다.
```python
with tf.Session() as sess:
    sess.run(iterator.initializer)
    image, label = sess.run([image_stacked, label_stacked])
```

참고 : https://medium.com/trackin-datalabs/input-data-tf-data-으로-batch-만들기-1c96f17c3696