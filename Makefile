main: evaluate_object_bev.cpp
	g++ -std=c++11 -o evaluate_object_bev_small_object evaluate_object_bev_small_object.cpp -lboost_system -lboost_filesystem
	g++ -std=c++11 -o evaluate_object_bev evaluate_object_bev.cpp -lboost_system -lboost_filesystem
.PHONY:
	clean

clean:
	rm -f evaluate_object_bev evaluate_object_bev_small_object
	
