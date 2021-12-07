import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]
x_ph = tf.placeholder(tf.float32, shape=[None])
y_ph = tf.placeholder(tf.float32, shape=[None])

# H(x) = Wx + b를 추측하는 과정
# W와 b를 1차원 변수로 랜덤 초기화
# tf.Variable로 생성하면 텐서플로우가 실행하면서 자동으로 업데이트하는 값

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
W_ph = tf.Variable(tf.random_normal([1]), name = 'weight')
b_ph = tf.Variable(tf.random_normal([1]), name = 'bias')
hypothesis = x_train * W + b
hypothesis_ph = x_ph * W_ph + b_ph

#cost(loss) function 생성
#reduce_mean 함수는 인자의 값을 평균 내주는 것
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost_ph = tf.reduce_mean(tf.square(hypothesis_ph - y_ph))

#cost(차이)를 Gradient descent로 줄이기
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
train_ph = optimizer.minimize(cost_ph)

#graph 실행을 위한 세션 생성, 전역 변수 초기화기 생성
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#세션 실행
for step in range(2001):
	sess.run(train) #그래프 전체 실행
	if step % 20 == 0: #20번에 한 번씩 출력
			print(step, sess.run(cost), sess.run(W), sess.run(b))

sess_ph = tf.Session()
sess_ph.run(tf.global_variables_initializer())

for step in range(2001):
	cost_val, W_val, b_val, _ = sess_ph.run([cost_ph, W_ph, b_ph, train_ph],
				    feed_dict = {x_ph: [1,2,3,4,5], y_ph:[2.1, 3.1, 4.1, 5.1, 6.1]})) 
	if step % 20 == 0: #20번에 한 번씩 출력
			print(step, cost_val, W_val, b_val)
