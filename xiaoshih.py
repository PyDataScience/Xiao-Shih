class SQS:
	def __init__(self, wv_model_path, keywords):
		from gensim.models import word2vec
		self.wv_model = word2vec.Word2Vec.load(wv_model_path)
		self.keywords = keywords

	def text2ngram(self, text, n):
		import jieba
		import re

		# tokenizing text
		word_list = jieba.cut(text.strip(), cut_all=False)
		word_list = list(word_list)
		
		# removing punctuations
		punctuation = set()
		with open ('corpus/punctuation_zh_tw.txt', 'r') as f:
			for line in f.readlines():
				punctuation.add(line.strip())
		with open ('corpus/punctuation_en.txt', 'r') as f:
			for line in f.readlines():
				punctuation.add(line.strip())
		punctuation.update({'\t', '\n', '\xa0', ' ', '　'})
		word_list = [word for word in word_list if word not in punctuation]

		# performing ngram algorithm
		ch_pattern = '[\u4e00-\u9fa5]+'
		ngram_list = []
		for i in range(len(word_list)):
			ngram = ''
			for j in range(n):
				if(i+j < len(word_list)):
					# combining Chinese words without any space
					if(re.match(ch_pattern, ngram)!= None and re.match(ch_pattern, word_list[i+j])!= None):
						ngram = ngram.strip() + word_list[i+j] + ' '
					else:
						ngram += word_list[i+j] + ' '
					ngram_list.append(ngram.strip())
		
		# removing stopwords
		stopwords = open("corpus/stopwords.txt").read()
		word_list = [word for word in word_list if word not in stopwords]

		# preserving words which present in the keyword list
		word_list = [word for word in word_list if word in self.keywords]

		return word_list

	def ngram2vsm(self, tokens):
		vsm = dict()
		for token in tokens:
			if(token not in vsm.keys()):
				vsm.update({token:1})
			else:
				vsm.update({token:vsm[token]+1})
		return vsm

	def spreading_similarity(self, depth, tokens, topn, vsm):
		if(depth < 0):
			print('The value of depth cannot be negative.')
			return {}
		elif(depth == 0):
			return vsm
		else:
			for token in tokens:
				vsm.update({token:vsm[token]+1})
				if(token in self.wv_model.wv.vocab.keys()):
					similar_tokens = self.wv_model.wv.most_similar(token, topn=topn)
					for keyword, similarity in similar_tokens:
						if(keyword in vsm.keys()):
							vsm.update({keyword:vsm[keyword]+similarity})
						else:
							vsm.update({keyword:similarity})
				else:
					continue
			tokens = list(vsm.keys())
			vsm = self.spreading_similarity(depth=depth-1, tokens=tokens, topn=topn, vsm=vsm)
			return vsm

	def cosine_similarity(self, vsm1, vsm2):
		len1 = 0
		len2 = 0
		matrix_mat = 0

		for value in vsm1.values():
			len1 += value**2
		len1 = len1**(1/2)
		for value in vsm2.values():
			len2 += value**2
		len2 = len2**(1/2)
		for key1 in vsm1.keys():
			for key2 in vsm2.keys():
				if(key1==key2):
					matrix_mat += vsm1[key1]*vsm2[key2]
		# handling division by zero
		try:
			return matrix_mat/(len1*len2)
		except:
			return 0

	def BERT_similarity(self, text1, text2, bert_model_path):
		import os
		import pandas as pd

		pd.DataFrame({'is_deplicated':[0],'Q1_id':[0],'Q2_id':[0],'Q1':[text1],'Q2':[text2]}).to_csv(bert_model_path +'/bert-master/bert_model_output/test.tsv',sep='\t',index=False,encoding='utf8')

		command = 'python '+ bert_model_path + '/bert-master/run_classifier.py \
					--task_name=selfsim \
					--do_predict=true \
					--data_dir='+ bert_model_path + '/bert-master/bert_model_output/ \
					--vocab_file=bert_model/multi_cased_L-12_H-768_A-12/vocab.txt \
					--bert_config_file=bert_model/multi_cased_L-12_H-768_A-12/bert_config.json \
					--init_checkpoint=' + bert_model_path + '/bert-master/bert_model_output/  \
					--max_seq_length=128 \
					--output_dir='+ bert_model_path + '/bert-master/bert_model_output/'
		os.system(command)

		df_bert_sim = pd.read_csv(bert_model_path + '/bert-master/bert_model_output/test_results.tsv',sep='\t', header=None)
		BERT_similarity = df_bert_sim.loc[0,1]

		return BERT_similarity

class QA:
	def __init__(self, wv_model_path, QA_model_path, bert_model_path, keywords, ml_features_path):
		from gensim.models import word2vec
		from sklearn.externals import joblib
		import pandas as pd

		self.wv_model_path = wv_model_path
		self.QA_model_path = QA_model_path
		self.bert_model_path = bert_model_path
		self.wv_model = word2vec.Word2Vec.load(wv_model_path)
		self.QA_model = joblib.load(QA_model_path)
		self.keywords = keywords
		self.df_ml = pd.read_pickle(ml_features_path)

	def answer_prediction(self, new_question, archived_question, answerer):
		import pandas as pd
		import numpy as np
		import re

		# computing question similarity by SQS
		sqs = SQS(wv_model_path=self.wv_model_path, keywords=self.keywords)
		tokens1 = sqs.text2ngram(text=new_question, n=5)
		vsm1 = sqs.ngram2vsm(tokens1)
		vsm1 = sqs.spreading_similarity(depth=1, tokens=tokens1, topn=10, vsm=vsm1)

		tokens2 = sqs.text2ngram(text=archived_question, n=5)
		vsm2 = sqs.ngram2vsm(tokens2)
		vsm2 = sqs.spreading_similarity(depth=1, tokens=tokens2, topn=10, vsm=vsm2)

		# feature 1: question similarity by SQS
		SQS_similarity = sqs.cosine_similarity(vsm1=vsm1, vsm2=vsm2)
		self.df_ml.loc[0, 'SQS'] = SQS_similarity

		# using the same threshold (0.7) while training Xiao-Shih
		if(SQS_similarity < 0.7):
			return 0
		
		# feature 2: question similarity by BERT
		self.df_ml.loc[0, 'BERT_Similarity'] = sqs.BERT_similarity(new_question, archived_question, self.bert_model_path)

		# feature 3: number of keywords in the new question
		self.df_ml.loc[0, 'q_num_keywords'] = len(set(tokens1))

		# feature 4: keyword rate of the new question
		if(len(new_question)!=0):
			self.df_ml.loc[0, 'keyword_rate'] = sum([len(token) for token in tokens1])/len(new_question)
		else:
			self.df_ml.loc[0, 'keyword_rate'] = 0

		# feature 5: one-hot encoded keywords of the new question
		other_features = ['SQS','BERT_Similarity','q_num_keywords', 'forum_question_nkws', 'keyword_rate',
						  'answerer_student', 'answerer_instructor', 'answerer_stackoverflow']
		for keyword in list(self.df_ml.drop(other_features, axis=1).columns):
			self.df_ml.loc[0, keyword] = 0
		for keyword in list(self.df_ml.drop(other_features, axis=1).columns):
			for token in tokens1:
				if(keyword == token):
					self.df_ml.loc[0, keyword] += 1

		# feature 6: number of keywords in the archived question
		self.df_ml.loc[0, 'forum_question_nkws'] = len(set(tokens2))

		# feature 7: the answerer is an instructor
		if(answerer == 'instructor'):
			self.df_ml.loc[0, 'answerer_instructor'] = 1
		else:
			self.df_ml.loc[0, 'answerer_instructor'] = 0

		# feature 8: the answerer is a student
		if(answerer == 'student'):
			self.df_ml.loc[0, 'answerer_student'] = 1
		else:
			self.df_ml.loc[0, 'answerer_student'] = 0

		# feature 9: the answerer is from stackoverflow
		if(answerer == 'stackoverflow'):
			self.df_ml.loc[0, 'answerer_stackoverflow'] = 1
		else:
			self.df_ml.loc[0, 'answerer_stackoverflow'] = 0

		# predicting whether the new question and the archived_question are duplicates or not
		prediction = pd.Series(self.QA_model.predict(self.df_ml))[0]

		return prediction