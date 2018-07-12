import os
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pdb

if __name__ == "__main__":
    # write the result into file
    label_dir = "/mnt/data2/test_xuetao/LidarAnnotation_old/label_2" #gt
    det_dir = "/mnt/data2/test_xuetao/LidarAnnotation_old/det_result" #finetune detection

    nofinetune_dir = "/mnt/data2/test_xuetao/LidarAnnotation_old/data_nofinetune"
    
    given_threshold = 0.15  # confidence

    for iter_num in range(1000, 15001, 1000):
        det_dir_used = det_dir + "_{}".format(iter_num)
        if os.path.isdir(det_dir_used):
            subprocess.call(["./run_eval.sh", 
                         str(label_dir),
                         str(det_dir_used),
                         str("true"), str(given_threshold)]) # for detection finetuned
        else:
            print("There is no correct file" 
                    "name or folder name, please check the argument")

    subprocess.call(["./run_eval.sh", 
                     str(label_dir),
                     str(nofinetune_dir),
                     str("false"), str(given_threshold)]) # no finetuned


    car_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/curve_car.txt")
    car_nofinetune_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/curve_nofinetune_car.txt")
    PC_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/curve_pedestrian.txt")
    PC_nofinetune_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/curve_nofinetune_pedestrian.txt")
    Truck_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/curve_truck.txt")
    Truck_nofinetune_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/curve_nofinetune_truck.txt")



    #pdb.set_trace()
    #car
    plt.text(0.1, 0.95, 'Car => Recall: Blue   Precision:Red', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(range(1, car_curve.shape[0]+1), car_curve[:, 0], 'ro')
    plt.plot(range(1, car_curve.shape[0]+1), car_curve[:, 1], 'b*')

    plt.plot([0, 20], [car_nofinetune_curve[0], car_nofinetune_curve[0]], 'r-')
    plt.plot([0, 20], [car_nofinetune_curve[1], car_nofinetune_curve[1]], 'b-')

    plt.axis([0, 20, 0, 1.0])
    plt.xlabel('iterations')
    plt.ylabel('Precision and Recall')
    plt.savefig("/mnt/data2/test_xuetao/LidarAnnotation_old/curvePlot_car.png")
    plt.close()

    #ped
    plt.text(0.1, 0.95, 'Pedestrian/Cyclist => Recall: Blue   Precision:Red', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(range(1, PC_curve.shape[0]+1), PC_curve[:, 0], 'ro')
    plt.plot(range(1, PC_curve.shape[0]+1), PC_curve[:, 1], 'b*')

    plt.plot([0, 20], [PC_nofinetune_curve[0], PC_nofinetune_curve[0]], 'r-')
    plt.plot([0, 20], [PC_nofinetune_curve[1], PC_nofinetune_curve[1]], 'b-')

    plt.axis([0, 20, 0, 1.0])
    plt.xlabel('iterations')
    plt.ylabel('Precision and Recall')
    plt.savefig("/mnt/data2/test_xuetao/LidarAnnotation_old/curvePlot_ped_cyc.png")
    plt.close()


    #Truck
    plt.text(0.1, 0.95, 'Truck => Recall: Blue   Precision:Red', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(range(1, Truck_curve.shape[0]+1), Truck_curve[:, 0], 'ro')
    plt.plot(range(1, Truck_curve.shape[0]+1), Truck_curve[:, 1], 'b*')

    plt.plot([0, 20], [Truck_nofinetune_curve[0], Truck_nofinetune_curve[0]], 'r-')
    plt.plot([0, 20], [Truck_nofinetune_curve[1], Truck_nofinetune_curve[1]], 'b-')

    plt.axis([0, 20, 0, 1.0])
    plt.xlabel('iterations')
    plt.ylabel('Precision and Recall')
    plt.savefig("/mnt/data2/test_xuetao/LidarAnnotation_old/curvePlot_truck.png")



    
