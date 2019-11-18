import json
import random
#the key for image dictionary is the image id number
#the value is a dictionary with round_id as key 
#and answer index as value
image_dictionary = {}
result_dictionary = {}
threshold = 10
ranks_filename = "ranks_0.1_data_faster_rcnn.json"
error_outputname = "error_output_0.1.json"

def parse_answer(answer_filename):
	with open(answer_filename) as answer_file:
		data = json.load(answer_file)['data']
		dialogs = data['dialogs']
		global questions_data 
		questions_data = data['questions']
		global answers_data
		answers_data = data['answers']
		for image in dialogs:
			#COCO image id
			image_id = image['image_id']
			round_id = 0
			for dialog in image['dialog']:
				#round_id
				round_id += 1
				#index of `answer` in `answer_options`
				gt_idx = dialog['gt_index']
				question_idx = dialog['question']
				answer_options = dialog['answer_options']
				#insert dialog info to dictionary
				if image_id in image_dictionary:
					dictionary = image_dictionary[image_id]
					dictionary[round_id] = [gt_idx,question_idx,answer_options]

					image_dictionary[image_id] = dictionary
				else:
					dictionary = {}
					dictionary[round_id] =[gt_idx,question_idx,answer_options]
					image_dictionary[image_id] = dictionary
	# for image_id,dictionary in image_dictionary.items():
	# 	print(image_id)
	# 	for k,v in dictionary.items():
	# 		print(k)
	# 		print(v)
def parse_result(result_filename):
	error_output = open(error_outputname,"w+")
	error_output.write("{\"errors\": [")
	with open(result_filename) as result_file:
		for data in result_file:
			data = "{\"result\":"+ data
			json_data = json.loads(data)
	result = json_data['result']
	total_error = 0
	for image in result:
		#COCO image id
		image_id = image['image_id']
		round_id = image['round_id']
		ranks = image['ranks']
		gt_idx = image_dictionary[image_id][round_id][0]
		question_idx = image_dictionary[image_id][round_id][1]
		answer_options = image_dictionary[image_id][round_id][2]
		best_answer_idx_1 = answer_options[ranks.index(1)]
		best_answer_idx_2 = answer_options[ranks.index(2)]
		best_answer_idx_3 = answer_options[ranks.index(3)]
		correct_answer_idx = answer_options[gt_idx]
		if ranks[gt_idx] > 10:
			total_error += 1
			#output to file
			error_output.write("{\"image_id\": "+ "\""+str(image_id) + "\""+ ",")
			error_output.write("\"questions\": "+ "\""+str(questions_data[question_idx]) + "\""+ ",")
			error_output.write("\"correct_answer\": " + "\""+ str(answers_data[correct_answer_idx])+ "\"" + ",")
			error_output.write("\"predict_answer\": " + "\" rank1: "+ str(answers_data[best_answer_idx_1]) + " rank2: "+ str(answers_data[best_answer_idx_2]) + " rank3: "+ str(answers_data[best_answer_idx_3]) + "\""+ "},\n")

	error_output.write("]}")
	error_output.close()

def preprocess_result(result_filename):
	f = open(ranks_filename,"a+")
	f.write("}")
	f.close()


if __name__ == '__main__':
	#parse val answer
	parse_answer("visdial_1.0_val.json")
	#run preprocess once then comment out:
	#preprocess_result(ranks_filename)
	#parse result
	parse_result(ranks_filename)
	index_list = random.sample(range(1,4770), 100)
	print(index_list)
	small_error_output = open("error_output_small.json","w+")
	with open(error_outputname) as error_output:
		lines = error_output.readlines()
		for i in index_list:
			small_error_output.write(lines[i])



		

