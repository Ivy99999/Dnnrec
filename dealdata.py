#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ivynie

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# print(tf.__version__)
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)

file_path = "data"
# print(os.getcwd())
# print(os.listdir(os.getcwd()))
# for root, dirs, files in os.walk(file_path):
#     print(root, dirs, files)

# %%time
# 读取用户数据
# 用户ID、性别、年龄、职业ID和邮编
users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table(os.path.join(file_path, "ml-1m/users.dat"), names=users_title, sep="::")
# print(users.head())
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table(os.path.join(file_path, "ml-1m/movies.dat"), names=movies_title, sep="::")
# print(movies.head())
# %%time
# 评分数据
# 用户ID、电影ID、评分和时间戳
ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
ratings = pd.read_table(os.path.join(file_path, "ml-1m/ratings.dat"), names=ratings_title, sep='::')
# print(ratings.head())
# 性别字典
gender_map = {"F": 0, "M": 1}

# 年龄字典
age_map = {value: id for id, value in enumerate(set(users["Age"]))}
# print(age_map)
# 进行浅拷贝
movies_source = movies.copy()


# # 删除年份(1996)
# s = "Grumpier Old Men (1995)"
# r_s = re.sub("\(\d+\)", "", s).strip()


def data_process(users, movies, ratings):
    users["Gender"] = users["Gender"].map(gender_map)
    users["Age"] = users["Age"].map(age_map)
    # print(users["Age"])
    movies["Title"] = movies["Title"].map(lambda x: re.sub("\(\d+\)", "", x).strip())

    # print(movies["Title"])
    def seq_process(df, col, sep):
        '''
        :param df: 传入dataframe
        :param col: 传入列名
        :param sep: 传入分隔符
        :return: 返回等长的映射id序列
        '''
        col_set = set()
        col_max_len = 0
        source_sep_list = []
        for val in df[col].values:
            sep_l = val.split(sep)
            # print(sep_l)
            col_set.update(sep_l)
            source_sep_list.append(sep_l)
            if len(sep_l) > col_max_len:
                col_max_len = len(sep_l)
        # 长度不足时，增加<PAD>
        col_set.add('<PAD>')
        col2id = {value: id for id, value in enumerate(col_set)}
        # print(col2id)
        dest_sep_list = [[col2id[t] for t in l] + [col2id["<PAD>"]] * (col_max_len - len(l)) for l in source_sep_list]
        df[col] = dest_sep_list
        # print(col + "列的最大长度序列为：", col_max_len)
        # print(col + "列的字典表为：", col2id)
        return df, col2id, col_max_len

    movies, title2id, title_maxlen = seq_process(movies, "Title", " ")
    movies, genre2id, genre_maxlen = seq_process(movies, "Genres", "|")

    data = pd.merge(pd.merge(ratings, users), movies)
    return data, title2id, title_maxlen, genre2id, genre_maxlen


data, title2id, title_maxlen, genre2id, genre_maxlen = data_process(users, movies, ratings)
# print(data.head())
# print("user number: ", len(users))
# print("movie number: ", len(movies))

# 划分训练集和测试集， 同时生成batch data
train, test = train_test_split(data, test_size=0.2, random_state=123)


# 或者
# train=data.sample(frac=0.8,random_state=200) #random state is a seed value
# test=data.drop(train.index)
def get_batches(Xs, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end]


batch_data = next(get_batches(train, 3))
print(batch_data)

# 开始构建模型
# 统计各个特征的数目
print(data.columns)
# Index(['UserID', 'MovieID', 'Rating', 'timestamps', 'Gender', 'Age',
# 'OccupationID', 'Zip-code', 'Title', 'Genres'], dtype='object')
# user info
vocab_uid = max(data["UserID"].unique()) + 1
vocab_gender = max(data["Gender"].unique()) + 1
vocab_age = max(data["Age"].unique()) + 1
vocab_job = max(data["OccupationID"].unique()) + 1

# movie info
vocab_mid = max(data["MovieID"].unique()) + 1

vocab_title = len(title2id)
vocab_genre = len(genre2id)
print(vocab_title, vocab_genre)
print(title_maxlen, genre_maxlen)

# 参数设置
emb_dim = 128
hidden_size = 256
genre_f = "sum"  # (or mean)
filter_sizes = [2, 3, 4, 5]  # 滑动2,3,4,5个单词
num_filters = 8  # 卷积核数
dropout_keep_prob = 0.5
lr = 0.001
num_epochs = 5
batch_size = 256
display_steps = 600


def get_inputs():
    # 定义placeholder
    with tf.name_scope("input_placeholder"):
        uid = tf.placeholder(tf.int32, shape=[None, 1], name="uid")
        gender = tf.placeholder(tf.int32, shape=[None, 1], name="user_gender")
        age = tf.placeholder(tf.int32, shape=[None, 1], name="user_age")
        job = tf.placeholder(tf.int32, shape=[None, 1], name="user_job")
        mid = tf.placeholder(tf.int32, shape=[None, 1], name="mid")
        title = tf.placeholder(tf.int32, shape=[None, 15], name="movie_title")
        genre = tf.placeholder(tf.int32, shape=[None, 6], name="movie_genre")
        target = tf.placeholder(tf.float32, shape=[None, 1], name="ratings")
    return uid, gender, age, job, mid, title, genre, target


def get_user_embedding(uid, gender, age, job):
    # 定义用户的embedding矩阵
    with tf.name_scope("u_embedding"):
        uid_embedding = tf.Variable(tf.random_normal([vocab_uid, emb_dim], 0, 1), name="user_embedding")
        uid_embed = tf.nn.embedding_lookup(uid_embedding, uid, name="user_embed")
        gender_embedding = tf.Variable(tf.random_normal([vocab_gender, emb_dim // 2], 0, 1), name="gender_embedding")
        gender_embed = tf.nn.embedding_lookup(gender_embedding, gender, name="gender_embed")
        age_embedding = tf.Variable(tf.random_normal([vocab_age, emb_dim // 2], 0, 1), name="age_embedding")
        age_embed = tf.nn.embedding_lookup(age_embedding, age, name="age_embed")
        job_embedding = tf.Variable(tf.random_normal([vocab_job, emb_dim // 2], 0, 1), name="job_embedding")
        job_embed = tf.nn.embedding_lookup(job_embedding, job, name="job_embed")
    return uid_embedding, uid_embed, gender_embedding, gender_embed, age_embedding, age_embed, job_embedding, job_embed


def user_nn(uid_embed, gender_embed, age_embed, job_embed):
    # 定义用户的context向量
    with tf.name_scope("user_nn"):
        # uid_embed: (batch_size, seq_len, emb_dim)
        uid_fc = tf.layers.dense(uid_embed, emb_dim, activation=tf.nn.relu)
        gender_fc = tf.layers.dense(gender_embed, emb_dim, activation=tf.nn.relu)
        age_fc = tf.layers.dense(age_embed, emb_dim, activation=tf.nn.relu)
        job_fc = tf.layers.dense(job_embed, emb_dim, activation=tf.nn.relu)

        # 对上述数据进行拼接
        u_cat = tf.concat([uid_fc, gender_fc, age_fc, job_fc], axis=-1)
        #         u_cat = tf.nn.tanh(tf.layers.dense(u_cat, hidden_size))
        u_cat = tf.layers.dense(u_cat, hidden_size, activation=tf.nn.tanh)
        uinfo = tf.reshape(u_cat, [-1, hidden_size])
    return uinfo


def movie_nn(mid, genre, title):
    with tf.name_scope("m_embedding"):
        mid_embedding = tf.Variable(tf.random_normal([vocab_mid, emb_dim], 0, 1))
        mid_embed = tf.nn.embedding_lookup(mid_embedding, mid)
        genre_embedding = tf.Variable(tf.random_normal([vocab_title, emb_dim // 2], 0, 1))
        genre_embed = tf.nn.embedding_lookup(genre_embedding, genre)
        if genre_f == "sum":
            genre_embed = tf.reduce_sum(genre_embed, axis=1, keepdims=True)
        elif genre_f == "mean":
            genre_embed = tf.reduce_mean(genre_embed, axis=1, keepdims=True)

        # 关于标题的cnn
        with tf.name_scope("title_cnn"):
            title_embedding = tf.Variable(tf.random_normal([vocab_title, emb_dim], 0, 1))
            title_embed = tf.nn.embedding_lookup(title_embedding, title)  # (batch_size, max_len, embed_size)
            # dim expand for cnn
            title_embed = tf.expand_dims(title_embed, -1)

            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                # 定义卷积核
                filter_shape = [filter_size, emb_dim, 1,
                                num_filters]  # [filter_height, filter_width, in_channels, channel_multiplier]
                filter_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                filter_b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

                conv = tf.nn.conv2d(title_embed, filter_W, [1, 1, 1, 1], padding="VALID")
                h = tf.nn.relu(tf.nn.bias_add(conv, filter_b))
                pooled = tf.nn.max_pool(h, ksize=[1, title_maxlen - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding="VALID")  # [batch_size, filter_height, filter_width, channel]
                # (batch_size, 1, 1, num_filters)
                pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, -1)
            num_filters_total = num_filters * len(filter_sizes)
            h_pool_flat = tf.reshape(h_pool, [-1, 1, num_filters_total])
            h_dropout = tf.nn.dropout(h_pool_flat, keep_prob=dropout_keep_prob)

    with tf.name_scope("m_nn"):
        mid_fc = tf.layers.dense(mid_embed, emb_dim, activation=tf.nn.relu)
        genre_fc = tf.layers.dense(genre_embed, emb_dim, activation=tf.nn.relu)
        m_cat = tf.concat([mid_fc, genre_fc, h_dropout], axis=-1)
        m_fc = tf.layers.dense(m_cat, hidden_size, activation=tf.nn.tanh)
        m_fc = tf.reshape(m_fc, [-1, hidden_size])
    return m_fc


# 构造计算图
tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    uid, gender, age, job, mid, title, genre, target = get_inputs()
    uid_embedding, uid_embed, gender_embedding, gender_embed, age_embedding, age_embed, job_embedding, job_embed = get_user_embedding(
        uid, gender, age, job)
    uinfo = user_nn(uid_embed, gender_embed, age_embed, job_embed)
    m_fc = movie_nn(mid, genre, title)
    inference = tf.reduce_sum(tf.multiply(uinfo, m_fc), axis=-1, keepdims=True, name="inference")
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(target, inference))
    #     loss_op = tf.reduce_mean(tf.square(target-inference))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss_op)


def get_batches(Xs, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end]


# # 训练网络
# model_path = os.path.join(file_path, "model/model.ckpt")  # 模型权重的保存地址
#
# with tf.Session(graph=train_graph) as sess:
#     # 创建summary来monitor loss_op
#     loss_summary = tf.summary.scalar("loss", loss_op)
#
#     # model和summaries文件目录
#     timestamp = str(int(time.time()))
#     out_dir = os.path.join(file_path, "runs", timestamp)
#     print("目录： ", out_dir)
#
#     # train op to write logs to Tensorboard
#     train_summary_dir = os.path.join(out_dir, "summaries", "train")
#     train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph=sess.graph)
#
#     # test op to write logs to Tensorboard
#     test_summary_dir = os.path.join(out_dir, "summaries", "test")
#     test_summary_writer = tf.summary.FileWriter(test_summary_dir, graph=sess.graph)
#
#     sess.run(tf.global_variables_initializer())
#     # Saver op to save and restore all the variables
#     saver = tf.train.Saver()
#
#     for epoch in range(num_epochs):
#         train, test = train_test_split(data, test_size=0.2)
#         train_batch_iterator = get_batches(train, batch_size)
#         test_batch_iterator = get_batches(test, batch_size)
#         for step in range(len(train) // batch_size):
#             batch_data = next(train_batch_iterator)
#             batchX, batchY = batch_data.drop("Rating", axis=1), batch_data["Rating"]
#             feed_dict = {
#                 uid: np.reshape(batchX["UserID"].values, [-1, 1]),
#                 gender: np.reshape(batchX["Gender"].values, [-1, 1]),
#                 age: np.reshape(batchX["Age"].values, [-1, 1]),
#                 job: np.reshape(batchX["OccupationID"].values, [-1, 1]),
#                 mid: np.reshape(batchX["MovieID"].values, [-1, 1]),
#                 title: np.array(batchX["Title"].values.tolist()),
#                 genre: np.array(batchX["Genres"].values.tolist()),
#                 target: np.reshape(batchY.values, [-1, 1]),
#             }
#             summaries, loss, _, rating_score, preidct = sess.run([loss_summary, loss_op, train_op, target, inference],
#                                                                  feed_dict=feed_dict)
#             # Write logs at every iteration(每一次迭代写一次数据)
#             train_summary_writer.add_summary(summaries, step)
#             #             print(rating_score[0])
#             #             print(preidct[0])
#
#             if step % display_steps == 0:
#                 now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 print("{}: Epoch {:>3} Batch {:>4}/{}  train_loss={:.3f}".format(now_time, epoch, step,
#                                                                                  len(train) // batch_size, loss))
#         # 进行测试
#         for step in range(len(test) // batch_size):
#             batch_data = next(test_batch_iterator)
#             batchX, batchY = batch_data.drop("Rating", axis=1), batch_data["Rating"]
#             feed_dict = {
#                 uid: np.reshape(batchX["UserID"].values, [-1, 1]),
#                 gender: np.reshape(batchX["Gender"].values, [-1, 1]),
#                 age: np.reshape(batchX["Age"].values, [-1, 1]),
#                 job: np.reshape(batchX["OccupationID"].values, [-1, 1]),
#                 mid: np.reshape(batchX["MovieID"].values, [-1, 1]),
#                 title: np.array(batchX["Title"].values.tolist()),
#                 genre: np.array(batchX["Genres"].values.tolist()),
#                 target: np.reshape(batchY.values, [-1, 1]),
#             }
#             summaries, loss, _ = sess.run([loss_summary, loss_op, train_op], feed_dict=feed_dict)
#             test_summary_writer.add_summary(summaries, step)
#
#             if step % display_steps == 0:
#                 now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 print("{}: Epoch {:>3} Batch {:>4}/{}  test_loss={:.3f}".format(now_time, epoch, step,
#                                                                                 len(test) // batch_size, loss))
#
#     # Save model weights to disk
#     saver.save(sess, save_path=model_path)
#     print("模型已经训练完成，同时已经保存到磁盘！")


def rating_movie(u_id, m_id):
    """
    指定用户和电影进行评分
    """
    with tf.Session() as sess:
        # 构造网络图
        saver = tf.train.import_meta_graph(os.path.join(file_path, "model/model.ckpt.meta"))
        # 加载参数
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(file_path, "model")))

        u_info = data[data["UserID"] == 234][["UserID", "Gender", "Age", "OccupationID"]].iloc[[0]]
        movie_info = data[data["MovieID"] == 1401][["MovieID", "Title", "Genres"]].iloc[[0]]
        # 访问图
        graph = tf.get_default_graph()
        #     for op in graph.get_operations():
        #         print(str(op.name))
        #     print(graph.get_tensor_by_name('input_placeholder/uid:0'))
        uid = graph.get_tensor_by_name('input_placeholder/uid:0')
        gender = graph.get_tensor_by_name('input_placeholder/user_gender:0')
        age = graph.get_tensor_by_name('input_placeholder/user_age:0')
        job = graph.get_tensor_by_name('input_placeholder/user_job:0')
        mid = graph.get_tensor_by_name('input_placeholder/mid:0')
        title = graph.get_tensor_by_name('input_placeholder/movie_title:0')
        genre = graph.get_tensor_by_name('input_placeholder/movie_genre:0')
        inference = graph.get_tensor_by_name('inference:0')

        feed_dict = {
            uid: np.reshape(u_info["UserID"].values, [-1, 1]),
            gender: np.reshape(u_info["Gender"].values, [-1, 1]),
            age: np.reshape(u_info["Age"].values, [-1, 1]),
            job: np.reshape(u_info["OccupationID"].values, [-1, 1]),
            mid: np.reshape(movie_info["MovieID"].values, [-1, 1]),
            title: np.array(movie_info["Title"].values.tolist()),
            genre: np.array(movie_info["Genres"].values.tolist()),
        }
        predict_score = sess.run([inference], feed_dict=feed_dict)
    return predict_score[0]


rating_movie(234, 1401)

# def save_weights():
#     """保存用户向量和电影向量结果"""
#     with tf.Session() as sess:
#         # 构造网络图
#         saver = tf.train.import_meta_graph(os.path.join(file_path, "model/model.ckpt.meta"))
#         # 加载参数
#         saver.restore(sess, tf.train.latest_checkpoint(os.path.join(file_path, "model")))
#
#         # 访问图
#         graph = tf.get_default_graph()
#         #         for op in graph.get_operations():
#         #             print(str(op.name))
#
#         uid = graph.get_tensor_by_name('input_placeholder/uid:0')
#         gender = graph.get_tensor_by_name('input_placeholder/user_gender:0')
#         age = graph.get_tensor_by_name('input_placeholder/user_age:0')
#         job = graph.get_tensor_by_name('input_placeholder/user_job:0')
#         mid = graph.get_tensor_by_name('input_placeholder/mid:0')
#         title = graph.get_tensor_by_name('input_placeholder/movie_title:0')
#         genre = graph.get_tensor_by_name('input_placeholder/movie_genre:0')
#         inference = graph.get_tensor_by_name('inference:0')
#         uinfo = graph.get_tensor_by_name('user_nn/Reshape:0')
#         m_fc = graph.get_tensor_by_name('m_nn/Reshape:0')
#         print(uinfo)
#         print(m_fc)
#
#         feed_dict = {
#             uid: np.reshape(users["UserID"].values, [-1, 1]),
#             gender: np.reshape(users["Gender"].values, [-1, 1]),
#             age: np.reshape(users["Age"].values, [-1, 1]),
#             job: np.reshape(users["OccupationID"].values, [-1, 1]),
#             mid: np.reshape(movies["MovieID"].values, [-1, 1]),
#             title: np.array(movies["Title"].values.tolist()),
#             genre: np.array(movies["Genres"].values.tolist()),
#         }
#         user_vec, movie_vec = sess.run([uinfo, m_fc], feed_dict=feed_dict)
#         np.savetxt(os.path.join(file_path, "data_vec", "user_vec.txt"), user_vec, fmt="%0.4f")
#         np.savetxt(os.path.join(file_path, "data_vec", "movie_vec.txt"), movie_vec, fmt="%0.4f")
# save_weights()
# 构建电影id和电影名称的字典表
id2title = pd.Series(movies_source["Title"].values, index=movies_source["MovieID"]).to_dict()


# def rec_similar_style(movie_id, topk=20):
#     """推荐相似电影"""
#     # 从txt文件读取数据
#     movie_vec = tf.constant(np.loadtxt(os.path.join(file_path, "data_vec", "movie_vec.txt"), dtype=np.float32))
#     # 对向量normalize
#     norm = tf.sqrt(tf.reduce_sum(tf.square(movie_vec), axis=-1, keepdims=True))
#     norm_movie_vec = movie_vec / norm
#     id_vec = tf.nn.embedding_lookup(norm_movie_vec, np.array([[movie_id]]))
#     id_vec = tf.reshape(id_vec, [-1, 256])
#     probs_similarity = tf.matmul(id_vec, tf.transpose(norm_movie_vec))
#     # 取前topk的电影
#     _, indices = tf.nn.top_k(probs_similarity, k=topk, sorted=True)
#     with tf.Session() as sess:
#         indices = sess.run([indices])
#     indices = indices[0].reshape(-1).tolist()[1:]
#     #     print(movies_source[movies_source["MovieID"].isin(indices)]["Title"])
#
#     #     rec_title = movies_source[movies_source["MovieID"].isin(indices)]["Title"].values.tolist()
#
#     print("您看的电影是：{}".format(id2title[movie_id]))
#     print("以下是给您的推荐：")
#     for indice in indices:
#         print(indice, ":", id2title[indice])
#
#
# indices = rec_similar_style(234, topk=5)


# 看过这个电影的人还喜欢什么电影
def view_also_view(movie_id, topk=20):
    """
    首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量。
    然后计算这几个人对所有电影的评分,选择每个人评分最高的电影作为推荐,同样加入了随机选择
    """
    movie_vec = tf.constant(np.loadtxt(os.path.join(file_path, "data_vec", "movie_vec.txt"), dtype=np.float32))
    user_vec = tf.constant(np.loadtxt(os.path.join(file_path, "data_vec", "user_vec.txt"), dtype=np.float32))
    id_vec = tf.nn.embedding_lookup(movie_vec, np.array([[movie_id]]))
    id_vec = tf.reshape(id_vec, [-1, 256])
    probs_similarity = tf.matmul(id_vec, tf.transpose(user_vec))
    _, indices = tf.nn.top_k(probs_similarity, k=topk, sorted=True)
    indices = tf.reshape(indices, [-1, 1])
    top_user_vec = tf.nn.embedding_lookup(user_vec, indices)
    top_user_vec = tf.reshape(top_user_vec, [-1, 256])

    sim_dist = tf.matmul(top_user_vec, tf.transpose(movie_vec))

    _, top_indices = tf.nn.top_k(sim_dist, k=2)
    top_indices = tf.reshape(top_indices, [-1])
    with tf.Session() as sess:
        indices = sess.run([top_indices])
    indices = indices[0].tolist()[1:]

    print("看过的电影是：{}".format(id2title[movie_id]))
    print("看过这个电影的人还喜欢什么电影：")
    for indice in indices:
        print(indice, ":", id2title[indice])


view_also_view(1401, topk=5)
