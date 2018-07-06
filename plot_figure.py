import subprocess
import matplotlib.pyplot as plt
import numpy as np


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


    car_curve = np.loadtxt("/home/saiclei/curve_car.txt")
    plt.plot([1, 2], car_curve[:, 0], 'ro')
    plt.plot([1, 2], car_curve[:, 1], 'b*')
    plt.axis([40, 160, 0, 0.03])
    plt.xlabel('iterations')
    plt.ylabel('Precision and Recall')
    plt.show()
