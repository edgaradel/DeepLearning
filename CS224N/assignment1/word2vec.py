import tensorflow as tf
import numpy as np
import math
import collections
import pickle as pkl
import jieba
import os

class word2vec():
    def __init__(self,
                 vocab_list=None,
                 embedding_size=200,
                 win_len=3,#单边窗口长
                 num_sampled=1000,
                 learning_rate=1.0,
                 logdir='/tmp/simple_word2vec',
                 model_path=None):
        # Get the basic parameters of the model.
        self.batch_size=None# The number of data in a batch, currently depends on the situation.
        if model_path!=None:
            self.load_model(model_path)
        else:
            #model parameters
            assert type(vocab_list)==list
            self.vocab_list=vocab_list
            self.vocab_size=vocab_list.__len__()
            self.embedding_size=embedding_size
            self.win_len=win_len
            self.num_sampled=num_sampled
            self.learning_rate=learning_rate
            self.logir=logdir

            self.word2id={} # word => id 的映射
            for i in range(self.vocab_size):
                self.word2id[self.vocab_list[i]]=i

            # train times
            self.train_words_num=0 # The number of word pairs for training.
            self.train_sents_num=0 # The number of sentences for training."
            self.train_times_num=0 # The number of training iterations (one iteration can include multiple sentences).

            # train loss records
            self.train_loss_records=collections.deque(maxlen=10)
            self.train_loss_k10=0

        self.build_graph()
        self.init_op()
        if model_path!=None:
            tf_model_path=os.path.join(model_path,'tf_vars')
            self.saver.restore(self.sess,tf_model_path)

    def init_op(self):
        self.sess=tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer=tf.summary.FileWriter(self.logir,self.sess.graph)

    def build_graph(self):
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.train_inputs=tf.placeholder(tf.int32,shape=[self.batch_size])
            self.train_labels=tf.placeholder(tf.int32,shape=[self.batch_size,1])
            self.embedding_dict=tf.Variable(
                tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0)
            )
            self.nce_weight=tf.Variable(tf.truncated_normal([self.vocab_size,self.embedding_size],
                                                            stddev=1.0/math.sqrt(self.embedding_size)))
            self.nce_biases=tf.Variable(tf.zeros([self.vocab_size]))

            # Vectorize the input sequence.
            embed=tf.nn.embedding_lookup(self.embedding_dict,self.train_inputs) # batch_size

            # Obtain NCE loss.
            self.loss=tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocab_size
                )
            )

            #TensorFlow related."
            tf.summary.scalar('loss',self.loss) #让tensorflow记录参数

            # Update gradients and embeddings based on NCE loss.
            self.train_op=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss) # 训练操作

            # Calculate the similarity with specified words.
            self.test_word_id=tf.placeholder(tf.int32,shape=[None])
            vec_l2_model=tf.sqrt( # Calculate the L2 norm of each word vector.
                tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True))
            avg_l2_model=tf.reduce_mean(vec_l2_model)
            tf.summary.scalar('avg_vec_model',avg_l2_model)

            self.normed_embedding=self.embedding_dict/vec_l2_model
            #self.embedding_dict=norm_vec # Normalize the embedding vectors.
            test_embed=tf.nn.embedding_lookup(self.normed_embedding,self.test_word_id)
            self.similarity=tf.matmul(test_embed,self.normed_embedding,transpose_b=True)

            # Variable initialization."
            self.init=tf.global_variables_initializer()

            self.merged_summary_op=tf.summary.merge_all()

            self.saver=tf.train.Saver()

    def train_by_sentence(self,input_sentence=[]):
        # input_sentence: [sub_sent1,sub_sent2,...]
        # Each sub_sent is a sequence of words
        sent_num=input_sentence.__len__()
        batch_inputs=[]
        batch_labels=[]
        for sent in input_sentence:
            for i in range(sent.__len__()):
                start=max(0,i-self.win_len)
                end=min(sent.__len__(),i+self.win_len+1)
                for index in range(start,end):
                    if index==i:
                        continue
                    else:
                        input_id=self.word2id.get(sent[i])
                        label_id=self.word2id.get(sent[index])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if(len(batch_inputs)==0):
            return
        batch_inputs=np.array(batch_inputs,dtype=np.int32)
        batch_labels=np.array(batch_labels,dtype=np.int32)
        batch_labels=np.reshape(batch_labels,[batch_labels.__len__(),1])

        feed_dict={
            self.train_inputs:batch_inputs,
            self.train_labels:batch_labels
        }
        _,loss_val,summary_str=self.sess.run([self.train_op,self.loss,self.merged_summary_op],feed_dict=feed_dict)

        # train loss
        self.train_loss_records.append(loss_val)
        self.train_loss_k10=np.mean(self.train_loss_records)
        if self.train_sents_num%1000==0:
            self.summary_writer.add_summary(summary_str,self.train_sents_num)
            print("{a} sentences dealed, loss: {b}"
                  .format(a=self.train_sents_num,b=self.train_loss_k10))

        # train times
        self.train_words_num+=batch_inputs.__len__()
        self.train_sents_num+=input_sentence.__len__()
        self.train_times_num+=1

    def cal_similarity(self,test_word_id_list,top_k=10):
        sim_matrix=self.sess.run(self.similarity,feed_dict={self.test_word_id:test_word_id_list})
        sim_mean=np.mean(sim_matrix)
        sim_var=np.mean(np.square(sim_matrix-sim_mean))
        test_words=[]
        near_words=[]
        for i in range(test_word_id_list.__len__()):
            test_words.append(self.vocab_list[test_word_id_list[i]])
            nearst_id=(-sim_matrix[i,:].argsort()[1:top_k+1])
            nearst_word=[self.vocab_list[x] for x in nearst_id]
            near_words.append(nearst_word)
        return test_words,near_words,sim_mean,sim_var

    def save_model(self,save_path):
        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Record various parameters of the model.
        model={}
        var_names=['vocab_size', #int
                   'vocab_list', #list
                   'learning_rate', #int
                   'word2id', #dict
                   'embedding_size', #int
                   'logdir', #str
                   'win_len', #int
                   'num_sampled', #int
                   'train_words_num', #int
                   'train_sents_num', #int
                   'train_times_num', #int
                   'train_loss_records', #int
                   'train_loss_k10', #int
                   ]
        for var in var_names:
            model[var]=eval('self.'+var)

        param_path=os.path.join(save_path,'params.pkl')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path,'wb') as file:
            pkl.dump(model,file)

        # Logging TensorFlow model.
        tf_path=os.path.join(save_path,'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess,tf_path)

    def load_model(self,model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        param_path=os.path.join(model_path,'params.pkl')
        with open(param_path,'rb') as file:
            model=pkl.load(file)
            self.vocab_list=model['vocab_list']
            self.vocab_size=model['vocab_size']
            self.logir=model['logdir']
            self.word2id=model['word2id']
            self.embedding_size=model['embedding_size']
            self.learning_rate=model['learning_rate']
            self.win_len=model['win_len']
            self.num_sampled=model['num_sampled']
            self.train_words_num=model['train_words_num']
            self.train_sents_num=model['train_sents_num']
            self.train_times_num=model['train_times_num']
            self.train_loss_records=model['train_loss_records']
            self.train_loss_k10=model['train_loss_k10']


if __name__ == '__main__':

    # Step 1: Read stop words.
    stop_words=[]
    with open('stop_words.txt','r',encoding='UTF-8') as file:
        line=file.readline()
        while line:
            stop_words.append(line[:-1])
            line=file.readline()
    stop_words=set(stop_words)
    print('Stop words read complete, {n} words in total'.format(n=len(stop_words)))

    # Step 2: Read text, preprocess, tokenize, and obtain vocabulary.
    raw_word_list=[]
    sentence_list=[]
    with open('280.txt',encoding='gbk') as file:
        line=file.readline()
        while line:
            while '\n' in line:
                line=line.replace('\n','')
            while ' ' in line:
                line=line.replace(' ','')
            if len(line)>0: #如果句子非空
                raw_words=list(jieba.cut(line,cut_all=False))
                dealed_words=[]
                for word in raw_words:
                    if word not in stop_words and word not in ['qingkan520','www','com','http']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line=file.readline()
        word_count=collections.Counter(raw_word_list)
        print('There are {n1} words in the text, {n2} unique words, selecting the top 30,000 words for the dictionary'.format(n1=n1, n2=n2))
              .format(n1=len(raw_word_list),n2=len(word_count)))
        word_count=word_count.most_common(30000)
        word_list=[x[0] for x in word_count]

        # Create model, train.
        w2v=word2vec(vocab_list=word_list, # Dictionary
                     embedding_size=200,
                     win_len=2,
                     learning_rate=1,
                     num_sampled=100,   # Number of negative samples.
                     logdir='/tmp/280')     # TensorBoard log directory."
        test_word = ['Xiao Yan', 'soul', 'flame', 'elder', 'elder', 'frown']
        test_id=[word_list.index(x) for x in test_word]
        num_steps=100000
        for i in range(num_steps):
            sent = sentence_list[i % len(sentence_list)]
            w2v.train_by_sentence([sent])













