#H(x) = Wx 라는 간단한 cost function를 가정하여 작성.
import tensorflow as tf
import matplotlib.pyplot as plt
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32) #W값을 바꿔가며 최소 cost 찾기 위함
hypothesis = X * W

#cost function 작성
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#그래프 그리기 위한 값 저장 리스트 생성
W_val = []
cost_val = []

for i in range(-30, 50):
	feed_W = i * 0.1  #W를 0.1 간격으로 -3~5 사이를 이동
	curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
	W_val.append(curr_W)
	cost_val.append(curr_cost)

#X축이 W, Y축이 cost인 그래프 시각화
plt.plot(W_val, cost_val)
plt.show()
