#GradientDescentOptimizer 함수를 사용한 코드
import tensorflow as tf
import matplotlib.pyplot as plt
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.0) #임의의 상수 삽입
hypothesis = X * W

#cost function 작성
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#GDO 함수로 W값을 자동 이동
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
	print(step, sess.run(W))
	sess.run(train)
