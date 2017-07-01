import os

SEQ_LEN = 40
ROOT_DIR = "data"
TEST_FILES = ["lstm.txt", "openai.csv", "cnn.csv", "multi.txt", "bidi.txt"]


data_source = DataSource(
    vocab=vocab,
    labeled_data_file="data/full_data.txt",
    test_data_file="data/test_data.txt",
    embedding_file=None,
    embedding_dim=None,
    seq_length=SEQ_LEN,
    embedding_type=None)

result = []

for i in range(0, 10000):
	result.append([0, 0])

for f in FILES:
	file_path = os.path.join(ROOT_DIR, f)
	file_pointer = open(file_path, 'r')
	lines = file_pointer.readlines()[1:]
	for i in range(0, len(lines)):
		l = lines[i].strip().split(',')
		if(int(l[1]) == 1):
			result[i][1] += 1
		else:
			result[i][0] += 1
	file_pointer.close()

max_sol = "max_sol.txt"
max_sol_path = os.path.join(ROOT_DIR, max_sol)
f = open(max_sol_path, 'w')
f.write("Id,Prediction\n")

for i in range(0, len(result)):
	if (result[i][0] > result[i][1]):
		f.write(str(i+1)+","+str("-1")+"\n")
	else:
		f.write(str(i+1)+","+str("1")+"\n")
