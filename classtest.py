# -*- coding:utf-8 -*-

class A(object):
    def __init__(self):
        self.a = "A"
        self.b = "B"

    def sayHello(self):
        print("This is A")

    def test(self, text):
        print("A test:", text)


a = A()
a.sayHello()


class B(A):
    def sayHello2(self):
        print(self.a, self.b)
        print("Hello, This is B")

    def sayHello(self):
        print("This is B")
        A.test(self, "From B")


b = B()
b.sayHello()
b.sayHello2()
