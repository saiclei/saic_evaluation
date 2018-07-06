import subprocess
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # write the result into file
    label_dir = "/home/saiclei/label_2"
    det_dir = "/home/saiclei/det_result_finetune"
    given_threshold = 0.1

    for iter_num in range(1000, 2001, 1000):
        det_dir_used = det_dir + "_{}".format(iter_num)
        subprocess.call(["./run_eval.sh", 
                     str(label_dir),
                     str(det_dir_used),
                     str(given_threshold)])



