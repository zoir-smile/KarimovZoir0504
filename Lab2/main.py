from random import shuffle

from layer import Layer
from data_reader import DataReader

NU = 000.1
H = 0.7


def main():
    reader = DataReader()
    train_data = reader.get_train_data()
    size = len(train_data[0].data)

    layer = Layer(size=size, h=H)
    layer.train(train_data=train_data, epochs=30, nu=NU)

    test_data = reader.get_test_data()
    shuffle(test_data)
    for item in test_data:
        print(f'This is output for ({item.number}). Answer is: {layer.predict(item)}')


if __name__ == '__main__':
    main()
