import mlx.core as mlx

a = mlx.array([1,2])
b = a
a += 1
print(a)
print(b)