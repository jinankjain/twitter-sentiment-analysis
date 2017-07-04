import os

RUN_SEQ_CONV = "python3 model.py --is_eval --emb_len=400" \
               " --embedding_type=word2vec" \
               " --ckpt_file=data/checkpoints/seq_conv_lstm_adam_ckpt-10000000-0.27.hdf5" \
               " --seq_conv2 --train_file=data/full_train.txt"

RUN_BIDI = "python3 model.py --is_eval --emb_len=200" \
           " --embedding_type=glove" \
           " --ckpt_file=data/checkpoints/bidi_lstm_adam_ckpt-15600000-0.26.hdf5" \
           " --bidi --train_file=data/full_train.txt"

RUN_GRU = "python3 model.py --is_eval --emb_len=200" \
          " --embedding_type=glove" \
          " --ckpt_file=data/checkpoints/gru_lstm_adam_ckpt-19200000-0.26.hdf5" \
          " --gru --train_file=data/full_train.txt"

RUN_SWISS_CHESSE = "python3 model.py --is_eval --emb_len=400" \
                   " --embedding_type=word2vec" \
                   " --ckpt_file=data/checkpoints/swisscheese_lstm_adam_ckpt-11400000-0.28.hdf5" \
                   " --seq_conv1 --train_file=data/full_train.txt"

# os.system(RUN_SEQ_CONV)
# os.system(RUN_BIDI)
# os.system(RUN_GRU)
# os.system(RUN_SWISS_CHESSE)

SEQ_LEN = 40
ROOT_DIR = "data"
FILES = [
    "bidi_test_output.txt", "gru_test_output.txt",
    "ensemble_test_outputs/conv_lstm_test_out_17400000.txt",
    "seq_conv2_test_output.txt", "seq_conv1_test_output.txt"]


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
