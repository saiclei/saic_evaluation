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
    

    ''' draw recall and precision curve '''
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
    plt.close()


    ''' draw mAP curve '''
    given_threshold = 0.0  # confidence 

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

    mAP_car_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curve_car.txt")
    mAP_car_nofinetune_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curve_nofinetune_car.txt")
    mAP_PC_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curve_pedestrian.txt")
    mAP_PC_nofinetune_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curve_nofinetune_pedestrian.txt")
    mAP_Truck_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curve_truck.txt")
    mAP_Truck_nofinetune_curve = np.loadtxt("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curve_nofinetune_truck.txt")


    #pdb.set_trace()
    #car
    
    plt.text(0.1, 99.5, 'Car => mAP: Red', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(range(1, len(mAP_car_curve)+1), mAP_car_curve[:], 'ro')
    #plt.plot(range(1, car_curve.shape[0]+1), car_curve[:, 1], 'b*')

    plt.plot([0, 20], [mAP_car_nofinetune_curve, mAP_car_nofinetune_curve], 'r-')
    #plt.plot([0, 20], [car_nofinetune_curve[1], car_nofinetune_curve[1]], 'b-')

    plt.axis([0, 20, 0, 100.0])
    plt.xlabel('iterations')
    plt.ylabel('mAP')
    plt.savefig("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curvePlot_car.png")
    plt.close()

    #ped
    plt.text(0.1, 99.5, 'Pedestrian/Cyclist => mAP: Red', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(range(1, len(mAP_PC_curve)+1), mAP_PC_curve[:], 'ro')
    #plt.plot(range(1, PC_curve.shape[0]+1), PC_curve[:, 1], 'b*')

    plt.plot([0, 20], [mAP_PC_nofinetune_curve, mAP_PC_nofinetune_curve], 'r-')
    #plt.plot([0, 20], [PC_nofinetune_curve[1], PC_nofinetune_curve[1]], 'b-')

    plt.axis([0, 20, 0, 100.0])
    plt.xlabel('iterations')
    plt.ylabel('mAP')
    plt.savefig("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curvePlot_ped_cyc.png")
    plt.close()


    #Truck
    plt.text(0.1, 99.5, 'Truck => mAP: Red', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(range(1, len(mAP_Truck_curve)+1), mAP_Truck_curve[:], 'ro')
    #plt.plot(range(1, Truck_curve.shape[0]+1), Truck_curve[:, 1], 'b*')

    plt.plot([0, 20], [mAP_Truck_nofinetune_curve, mAP_Truck_nofinetune_curve], 'r-')
    #plt.plot([0, 20], [Truck_nofinetune_curve[1], Truck_nofinetune_curve[1]], 'b-')

    plt.axis([0, 20, 0, 100.0])
    plt.xlabel('iterations')
    plt.ylabel('mAP')
    plt.savefig("/mnt/data2/test_xuetao/LidarAnnotation_old/mAP_curvePlot_truck.png")

    
