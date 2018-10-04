# numpy Practice
import numpy as np

a1 = np.array([1,2,3,4,5,6,7,8,9])
# print(a1, type(a1), a1.shape, a1.dtype, a1.size, np.sum(a1))

a2 = np.array([[1,2,3,4], [5,6,7,8]])
# print(a2, type(a2), a2.shape, a2.dtype, a2.size, np.sum(a2), np.sum(a2, axis=0), np.sum(a2, axis=1),
#       np.mean(a2), np.mean(a2, axis=0), "Sqrt:", np.sqrt(a2), "logarith:", np.log(a2))

t = np.log(a2)

# print(np.exp(t))

a3 = np.random.random((3,4))
# print(a3)
a3[2,3] = 100
# print(a3)

a3 = np.random.random_sample((3,4))
# print(a3, a3.shape)

a4 = np.arange(10, dtype=np.float64).reshape(2,5)
# print(a4)

a5 = np.linspace(1,10,6) # startpoint, endpoint, number of element
# print(a5)

a6 = np.zeros((3,7))
# print(a6, a6.size)

a6.fill(5)
# print(a6)

b1 = a6.reshape(a6.size)
# print(b1, b1.size)

b2 = a6.ravel()  # print at flatten
# print(b2, b2.size)

# print(a6)

t2 = np.vstack(a6)
# print(t2) # print at n-demenssion


c1 = np.arange(12).reshape(3,4)
# print(c1)

c2 = np.random.random(12).reshape(4,3)
# print(c2)

c3 = np.random.random(12).reshape(3,4)
# print(c3)

a2 = a1 * 2
# print(a1)
# print(a1 + a2, a1*a2)
# print(np.add(a1,a2))

# print(c3 + c1)

# print(c1 * c2)

d1 = np.arange(12)
d2 = np.arange(12).reshape(3,4)
d3 = np.arange(60).reshape(4,3,5)

print(d1)
print(d2)
print(d3)


