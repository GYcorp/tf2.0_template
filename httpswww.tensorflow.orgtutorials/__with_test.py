
class Sub:
    def run(self, x):
        print('Sub running...')
        return x * x

class A:
    def __enter__(self):
        print('__class A enter__')
        return Sub()

    def __exit__(self, type, value, traceback):
        print('__class A exit__')

with A() as sub:
    print(sub.run(10))

print(sub.run(100))