#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <string>
#include <assert.h>

#include <dirent.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>


BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)


typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > polygon;

using namespace std;

template<typename T>
void showVec(const T& container) {
    for (const auto& item : container) {
        cout << item << '\n';
    }
}

/*=======================================================================
static evaluation parameters
=======================================================================*/

// evaluated object classes
enum classes{car=0, pedestrian=1, truck=2};
const int num_class = 3;

// parameters varying per class
vector<string> class_names;
// the minimum overlap required for 2d evaluation on the image/ground plane and 3d evaluation7const double min_overlap[3] = {0.5, 0.5, 0.5};7
// // no. of recall steps that should be evaluated (discretized)
const double min_overlap[3] = {0.5, 0.2, 0.5};

double n_sample_pts = 41;
int n_recall_step = 1;         // 4
double d_sum_denominator = n_sample_pts; // 11

// initialize class names
void initglobals () {
  class_names.push_back("car");
  class_names.push_back("pedestrian");
  class_names.push_back("truck");
}

/*=======================================================================
data types for evaluation
=======================================================================*/

// holding data needed for precision-recall and precision-aos
struct tprdata {
  vector<double> v;           // detection score for computing score thresholds
  vector< pair<double, bool> > ap_vec;
  int        tp;          // true positives
  int        fp;          // false positives
  int        fn;          // false negatives
  tprdata () :
    tp(0), fp(0), fn(0) {}
};

// holding bounding boxes for ground truth and detections
struct tbox {
  string  type;     // object type as car, pedestrian or cyclist,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tbox (string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tgroundtruth {
  tbox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  double ry;
  double  t1, t2, t3;
  double h, w, l;
  tgroundtruth () :
    box(tbox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1) {}
  tgroundtruth (tbox box,double truncation,int32_t occlusion) :
    box(box),truncation(truncation),occlusion(occlusion) {}
  tgroundtruth (string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion) :
    box(tbox(type,x1,y1,x2,y2,alpha)),truncation(truncation),occlusion(occlusion) {}
};


// holding detection data
struct tdetection {
  tbox    box;    // object type, box, orientation
  double  thresh; // detection score
  double  ry;
  double  t1, t2, t3;
  double  h, w, l;
  tdetection ():
    box(tbox("invalid",-1,-1,-1,-1,-10)),thresh(-1000) {}
  tdetection (tbox box,double thresh) :
    box(box),thresh(thresh) {}
  tdetection (string type,double x1,double y1,double x2,double y2,double alpha,double thresh) :
    box(tbox(type,x1,y1,x2,y2,alpha)),thresh(thresh) {}
};

// All in Lidar


/*=======================================================================
functions to load detection and ground truth data once, save results
=======================================================================*/
vector<int32_t> indices;

vector<tdetection> loaddetections(string file_name, vector<bool> &eval_ground, bool &success) {

    // holds all detections (ignored detections are indicated by an index vector
    vector<tdetection> detections;
    FILE *fp = fopen(file_name.c_str(),"r");
    if (!fp) {
        success = false;
        return detections;
    }
    while (!feof(fp)) {
        tdetection d;
        double trash;
        char str[255];
        if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &trash, &trash, &d.box.alpha, &d.box.x1, &d.box.y1,
                   &d.box.x2, &d.box.y2, &d.h, &d.w, &d.l, &d.t1, &d.t2, &d.t3,
                   &d.ry, &d.thresh)==16) {

            // d.thresh = 1;
            d.box.type = str;
            if (!strcasecmp(str, "van"))
                d.box.type = "car";

            if (!strcasecmp(str, "cyclist"))
                d.box.type = "pedestrian";
            detections.push_back(d);


            // a class is only evaluated if it is detected at least once
            for (int c = 0; c < num_class; c++) {
		    // strcasecmp: compare strings without case sensitivity, <0: less, ==0 equal, >0: larger
                if (!strcasecmp(d.box.type.c_str(), class_names[c].c_str())) {
                    //if (!eval_ground[c] && d.t1 != -1000 && d.t3 != -1000 && d.w > 0 && d.l > 0)
                    eval_ground[c] = true;
                    break;
                }
            }
        }
    }
    fclose(fp);
    success = true;
    return detections;
}

vector<tgroundtruth> loadgroundtruth(string file_name,bool &success) {

    // holds all ground truth (ignored ground truth is indicated by an index vector
    vector<tgroundtruth> groundtruth;
    FILE *fp = fopen(file_name.c_str(),"r");
    if (!fp) {
        success = false;
        return groundtruth;
    }
    while (!feof(fp)) {
        tgroundtruth g;
        char str[255];
        if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &g.truncation, &g.occlusion, &g.box.alpha,
                   &g.box.x1,   &g.box.y1,     &g.box.x2,    &g.box.y2,
                   &g.h,      &g.w,        &g.l,       &g.t1,
                   &g.t2,      &g.t3,        &g.ry )==15) {
        g.box.type = str;
        if (!strcasecmp(str, "van"))
            g.box.type = "car";

        if (!strcasecmp(str, "cyclist"))
            g.box.type = "pedestrian";
        groundtruth.push_back(g);
        }
    }
    fclose(fp);
    success = true;
    return groundtruth;
}

void savestats (const vector<double> &precision,  FILE *fp_det) {

  // save precision to file
  if(precision.empty())
    return;
  for (int32_t i=0; i<precision.size(); i++)
    fprintf(fp_det,"%f ",precision[i]);
  fprintf(fp_det,"\n");

 }

/*=======================================================================
evaluation helper functions
=======================================================================*/

// compute polygon of an oriented bounding box
template <typename t>
polygon topolygon(const t& g) {
    using namespace boost::numeric::ublas;
    using namespace boost::geometry;
    matrix<double> mref(2, 2);
    mref(0, 0) = cos(g.ry); mref(0, 1) = sin(g.ry);
    mref(1, 0) = -sin(g.ry); mref(1, 1) = cos(g.ry);

    static int count = 0;
    matrix<double> corners(2, 4);
    double data[] = {g.l / 2, g.l / 2, -g.l / 2, -g.l / 2,
                     g.w / 2, -g.w / 2, -g.w / 2, g.w / 2};
    std::copy(data, data + 8, corners.data().begin());
    matrix<double> gc = prod(mref, corners);
    for (int i = 0; i < 4; ++i) {
        gc(0, i) += g.t1;
        gc(1, i) += g.t3;
    }

    double points[][2] = {{gc(0, 0), gc(1, 0)},{gc(0, 1), gc(1, 1)},{gc(0, 2), gc(1, 2)},{gc(0, 3), gc(1, 3)},{gc(0, 0), gc(1, 0)}};
    polygon poly;
    append(poly, points);
    return poly;
}

// measure overlap between bird's eye view bounding boxes, parametrized by (ry, l, w, tx, tz)
inline double groundboxoverlap(tdetection d, tgroundtruth g, int32_t criterion = -1) {
    using namespace boost::geometry;
    polygon gp = topolygon(g);
    polygon dp = topolygon(d);

    std::vector<polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double inter_area = in.empty() ? 0 : area(in.front());
    double union_area = area(un.front());
    double o;
    if(criterion==-1)     // union
        o = inter_area / union_area;
    else if(criterion==0) // bbox_a
        o = inter_area / area(dp);
    else if(criterion==1) // bbox_b
        o = inter_area / area(gp);


    /*    
	cout << "detection t1 is: " << d.t1 << " and t2 is: " << d.t2 << " and t3 is: " << d.t3
		 << " ground-truth t1 is: " << g.t1 << " and t2 is: " << g.t2 << " and t3 is: " << g.t3 
		 << " and the iou is: " << o << endl;
	*/
    return o;
}


vector<double> getThresholds(vector<double>& v, double n_groundtruth) {
    vector<double> t;

    sort(v.begin(), v.end(), greater<double>());

    double current_recall = 0;

    for (int i = 0; i < v.size(); ++i) {
        double l_recall, r_recall, recall;
        l_recall = (double)(i+1) / n_groundtruth;
        if (i < v.size() - 1)
            r_recall = (double)(i+2) / n_groundtruth;
        else
            r_recall = l_recall;

        if ( (r_recall - current_recall) < (current_recall - l_recall) && i < (v.size() - 1))
            continue;

        recall = l_recall;

        t.push_back(v[i]);
        current_recall += 1.0 / (n_sample_pts - 1.0);
    }
    return t;
}

void cleanData(classes current_class, const vector<tgroundtruth>& gt, const vector<tdetection>& det, vector<int>& ignored_gt,
               vector<int>& ignored_det, int& n_gt) {
    // cleanData for each image, ignored_gt and ignored_det is an empty vector corresponding to a single image
    // n_gt count in the same class
    for (const auto& gt_element : gt) {
        if (!strcasecmp(gt_element.box.type.c_str(), class_names[current_class].c_str())) {
            ignored_gt.push_back(0);
            n_gt++;
        } else
            ignored_gt.push_back(-1);
    }
    for (const auto& det_element : det) {
        if (!strcasecmp(det_element.box.type.c_str(), class_names[current_class].c_str())) {
            ignored_det.push_back(0);
        } else
            ignored_det.push_back(-1);
    }
}

tprdata computeStatistics(classes current_class, const vector<tgroundtruth>& gt, const vector<tdetection>& det,
                          const vector<int>& ignored_gt, const vector<int>& ignored_det,
                          bool compute_fp, double (*boxoverlap)(tdetection, tgroundtruth, int),
                          double thresh = 0) {
    tprdata stat = tprdata();
    const double no_detection = -1;
    vector<bool> assigned_detection(det.size(), false);
    vector<bool> ignored_threshold(det.size(), false);
    double epsilon = 0.0001;
    if (compute_fp) {
        for (int i = 0; i < det.size(); ++i) {
            if (det[i].thresh < thresh)
                ignored_threshold[i] = true;
        }
    }

    // Evaluate all ground truth boxes
    for (int i = 0; i < gt.size(); ++i) {
        if (ignored_gt[i] == -1)
            continue;
        
        int det_idx = -1;
        double valid_detection = no_detection;
        double max_overlap = 0;

        // Search for a possible detection
        bool assigned_ignored_det = false;
        for (int j = 0; j < det.size(); ++j) {
            if (ignored_det[j] == -1 || assigned_detection[j] || ignored_threshold[j])
                continue;
            
            // Find the maximum score for the candidates and get idx of respective detection
            double overlap = boxoverlap(det[j], gt[i], -1);
            if (!compute_fp && overlap > min_overlap[current_class] && det[j].thresh > valid_detection) {
                det_idx = j;
                valid_detection = det[j].thresh;
            } else if (compute_fp && overlap > min_overlap[current_class] &&
                       (overlap > max_overlap || assigned_ignored_det) && ignored_det[j] == 0) {
                max_overlap             = overlap;
                det_idx                 = j;
                valid_detection         = 1;
                assigned_ignored_det    = false;
            } else if (compute_fp && overlap > min_overlap[current_class] &&
                       valid_detection == no_detection && ignored_det[j] == 1) {
                det_idx                 = j;
                valid_detection         = 1;
                assigned_ignored_det    = true;
            }

        }

        // Compute tp, fp and fn
        if (abs(valid_detection - no_detection) < epsilon && ignored_gt[i] == 0) {
            stat.fn++;
        }else if (abs(valid_detection - no_detection) > epsilon ) {
            stat.tp++;
            stat.v.push_back(det[det_idx].thresh);
            stat.ap_vec.push_back(make_pair(det[det_idx].thresh, true));
            assigned_detection[det_idx] = true;
        }

    }

    if (compute_fp) {
        for (int i = 0; i < det.size(); ++i) {
            if (!(assigned_detection[i] || ignored_det[i] == -1 || ignored_threshold[i])) {
                stat.fp++;
                stat.ap_vec.push_back(make_pair(det[i].thresh, false));
            }
        }
    }

    return stat;

}

/*=======================================================================
 * Calculate the AP based on the rule after VOC 2010
=========================================================================*/
bool new_eval_class (FILE *fp_det, FILE *fp_ori,classes current_class,
                 const vector< vector<tgroundtruth> > &groundtruth,
                 const vector< vector<tdetection> > &detections,
                  double (*boxoverlap)(tdetection, tgroundtruth, int32_t),
                 vector<double>& recall,
                 vector<double>& thresholds,
                 vector<double>& precision) {
	assert(groundtruth.size() == detections.size());    // Make sure each image has a ground truth and detection respectively.
  	// init
  	int n_gt=0;                                     // total no. of gt (denominator of recall)
  	vector<double> v;                   // detection scores, evaluated for recall discretization
  	vector< vector<int> > ignored_gt, ignored_det;  // index of ignored gt detection for current class
    vector< pair<double, bool>> test_vec;
  	// for all test images do
  	for (int i=0; i<groundtruth.size(); i++){
    	// holds ignored ground truth, ignored detections and dontcare areas for current frame
    	vector<int> i_gt, i_det;

    	// only evaluate objects of current class 
    	cleanData(current_class, groundtruth[i], detections[i], i_gt, i_det, n_gt); 
    	ignored_gt.push_back(i_gt);
    	ignored_det.push_back(i_det);

    	// compute statistics to get recall values, tp/(tp+fn)
    	tprdata pr_tmp = tprdata();
    	pr_tmp = computeStatistics(current_class, groundtruth[i], detections[i], i_gt, i_det, true, boxoverlap, 0);

    	// add detection scores to vector over all images
    	for(int j=0; j<pr_tmp.v.size(); j++)
      	    v.push_back(pr_tmp.v[j]);         // all true positive

        for (int j = 0; j < pr_tmp.ap_vec.size(); j++) {
            test_vec.push_back(pr_tmp.ap_vec[j]);
        } 
  	}
    // Start to since VOC2010 methods.
    sort(test_vec.begin(),  test_vec.end(),
            [](const pair<double, bool>& a, const pair<double, bool>& b) {return a.first > b.first;});
    
    vector<tprdata> pr (v.size(), tprdata());


    int tp(0), fp(0);
	recall.push_back(0);
	precision.push_back(0);
	thresholds.push_back(1);
    for (int i = 0; i < test_vec.size(); ++i) {
        if (test_vec[i].second) {
            tp++;
            recall.push_back(double(tp)/n_gt);
            precision.push_back(double(tp) / double(tp+fp));
            thresholds.push_back(test_vec[i].first);
        } else {
            fp++;
        }
    }

		
	for (int i = 0; i < precision.size(); ++i) {
		precision[i] = *max_element(precision.begin() + i, precision.end());
	}
	
	precision.push_back(0);
	recall.push_back(recall.back());
	thresholds.push_back(0);
  	// save statisics and finish with success
  	savestats(precision, fp_det);
    return true;
}

/*=======================================================================
evaluate class-wise
=======================================================================*/
bool eval_class (FILE *fp_det, FILE *fp_ori,classes current_class,
                 const vector< vector<tgroundtruth> > &groundtruth,
                 const vector< vector<tdetection> > &detections,
                  double (*boxoverlap)(tdetection, tgroundtruth, int32_t),
                 vector<double> &precision) {
	assert(groundtruth.size() == detections.size());    // Make sure each image has a ground truth and detection respectively.
  	// init
  	int n_gt=0;                                     // total no. of gt (denominator of recall)
  	vector<double> v, thresholds;                   // detection scores, evaluated for recall discretization
  	vector< vector<int> > ignored_gt, ignored_det;  // index of ignored gt detection for current class

  	// for all test images do
  	for (int i=0; i<groundtruth.size(); i++){
    	// holds ignored ground truth, ignored detections and dontcare areas for current frame
    	vector<int> i_gt, i_det;

    	// only evaluate objects of current class 
    	cleanData(current_class, groundtruth[i], detections[i], i_gt, i_det, n_gt); 
    	ignored_gt.push_back(i_gt);
    	ignored_det.push_back(i_det);

    	// compute statistics to get recall values, tp/(tp+fn)
    	tprdata pr_tmp = tprdata();
    	pr_tmp = computeStatistics(current_class, groundtruth[i], detections[i], i_gt, i_det, false, boxoverlap, 0);

    	// add detection scores to vector over all images
    	for(int j=0; j<pr_tmp.v.size(); j++)
      	    v.push_back(pr_tmp.v[j]);
  	}
     
  	// get scores that must be evaluated for recall discretization
  	thresholds = getThresholds(v, n_gt);
  	// compute tp,fp,fn for relevant scores
  	vector<tprdata> pr(thresholds.size(), tprdata());
  	for (int i=0; i<groundtruth.size(); i++){

    	// for all scores/recall thresholds do:
    	for(int32_t t=0; t<thresholds.size(); t++){
      		tprdata tmp = tprdata();
      		tmp = computeStatistics(current_class, groundtruth[i], detections[i], 
                              ignored_gt[i], ignored_det[i], true, boxoverlap,
                              thresholds[t]);

      		// add no. of tp, fp, fn for current frame to total evaluation for current threshold
      		pr[t].tp += tmp.tp;
      		pr[t].fp += tmp.fp;
      		pr[t].fn += tmp.fn;
    	}
  	}

  	// compute recall, precision 
  	vector<double> recall;
  	precision.assign(n_sample_pts, 0);

  	double r=0;
  	for (int32_t i=0; i<thresholds.size(); i++){
    	r = pr[i].tp/(double)(pr[i].tp + pr[i].fn);
    	recall.push_back(r);
    	precision[i] = pr[i].tp/(double)(pr[i].tp + pr[i].fp);

  	}

  	// filter precision using max_{i..end}(precision)
	/*
  	for (int32_t i=0; i<thresholds.size(); i++){
    	precision[i] = *max_element(precision.begin()+i, precision.end());
  	}
	*/
  	// save statisics and finish with success
  	savestats(precision, fp_det);
    return true;
}


/*=======================================================================
evaluate class-wise
=======================================================================*/

bool eval_class_given_threshold (classes current_class,
                 const vector< vector<tgroundtruth> > &groundtruth,
                 const vector< vector<tdetection> > &detections,
                  double (*boxoverlap)(tdetection, tgroundtruth, int32_t), double given_threshold = 0.5) {

	assert(groundtruth.size() == detections.size());
  	// init
  	int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
  	vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/difficulty

  	// for all test images do
  	for (int32_t i=0; i<groundtruth.size(); i++){
    	// holds ignored ground truth, ignored detections and dontcare areas for current frame
    	vector<int32_t> i_gt, i_det;

    	// only evaluate objects of current class and ignore occluded, truncated objects
    	cleanData(current_class, groundtruth[i], detections[i], i_gt, i_det, n_gt);
    	ignored_gt.push_back(i_gt);
    	ignored_det.push_back(i_det);

    	// compute statistics to get recall values
    	tprdata pr_tmp = tprdata();

  	}
    
    double tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < groundtruth.size(); ++i) {
        tprdata tmp = tprdata();
        tmp = computeStatistics(current_class, groundtruth[i], detections[i], 
                              ignored_gt[i], ignored_det[i], true, boxoverlap,
                              given_threshold);
        tp += tmp.tp;
        fp += tmp.fp;
        fn += tmp.fn;
    }
    cout << "tp, fp and fn are: " << tp << ", " << fp << ", and " << fn << endl;
  	double recall = tp / (tp + fn);
    double precision = tp / (tp + fp);
    cout << "Current object type is: " << class_names[current_class] 
         << ", Given confidence socre thresold is: " << given_threshold << ", recall is: " 
         << recall << " and precision is: " << precision << endl;

    FILE *fp_curve = fopen(("/home/saiclei/curve_" + class_names[current_class] + ".txt").c_str(),"a+");
    fprintf(fp_curve, "%f %f\n", precision, recall);
    fclose(fp_curve);
   return true; 
}

// vals is precisions
void newSaveAndPlotPlots(string dir_name, string file_name, string obj_type, 
                         vector<double> recalls, vector<double> precisions, vector<double> thresholds) {
    assert(precisions.size() == recalls.size());
	
    char command[1024];
    FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(), "w");
    for (int i = 0; i < precisions.size(); ++i) {
        fprintf(fp, "%f %f %f\n", recalls[i], precisions[i], thresholds[i]);
    }
    fclose(fp);
    float sum = 0.0;

    for (int i = 0; i < recalls.size() - 1; ++i) {
        sum += (recalls[i+1] - recalls[i]) * precisions[i+1];
    }
    printf("%s ap: %f\n", file_name.c_str(), sum); 
    // create png + eps
    for (int j=0; j<2; j++) {
       
        // open file
        FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");
        // save gnuplot instructions
        if (j==0) {
            fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
            fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
        } else {
            fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
            fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
        }

        // set labels and ranges
        fprintf(fp,"set size ratio 0.7\n");
        fprintf(fp,"set xrange [0:1]\n");
        fprintf(fp,"set yrange [0:1]\n");
        fprintf(fp,"set xlabel \"Recall\"\n");
        fprintf(fp,"set ylabel \"Precision\"\n");
        obj_type[0] = toupper(obj_type[0]);
        fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

        // line width
        int   lw = 5;
        if (j==0) lw = 3;

        // plot error curve
        fprintf(fp,"plot ");
        //fprintf(fp,"\"%s.txt\" using 1:2 title 'PR curve', using 1:3 title 'thresholds' with lines ls 1 lw %d,",file_name.c_str(),lw);
        fprintf(fp,"\"%s.txt\" using 1:2 title 'PR curve' with lines ls 1 lw %d, '' using 1:3 title 'thresholds' with lines ls 1 lw %d lt -1",file_name.c_str(), lw, lw);
        // close file
        fclose(fp);

        // run gnuplot => create png + eps
        sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
        system(command);
    }
    
}

void saveandplotplots(string dir_name,string file_name,string obj_type,vector<double> vals){
    char command[1024];

    // save plot data to file
    FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
    for (int32_t i=0; i<(int)n_sample_pts; i++)
        fprintf(fp,"%f %f\n",(double)i/(n_sample_pts-1.0),vals[i]);
    fclose(fp);
    float sum = 0.0;
    for (int i = 0; i < vals.size(); i = i + n_recall_step)
	    sum += vals[i];

    printf("%s ap:%f\n", file_name.c_str(), sum / d_sum_denominator * 100);

    // create png + eps
    for (int j=0; j<2; j++) {
       
        // open file
        FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");
        // save gnuplot instructions
        if (j==0) {
            fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
            fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
        } else {
            fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
            fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
        }

        // set labels and ranges
        fprintf(fp,"set size ratio 0.7\n");
        fprintf(fp,"set xrange [0:1]\n");
        fprintf(fp,"set yrange [0:1]\n");
        fprintf(fp,"set xlabel \"Recall\"\n");
        fprintf(fp,"set ylabel \"Precision\"\n");
        obj_type[0] = toupper(obj_type[0]);
        fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

        // line width
        int   lw = 5;
        if (j==0) lw = 3;

        // plot error curve
        fprintf(fp,"plot ");
        fprintf(fp,"\"%s.txt\" using 1:2 title 'PR curve' with lines ls 1 lw %d,",file_name.c_str(),lw);

        // close file
        fclose(fp);

        // run gnuplot => create png + eps
        sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
        system(command);
    }
}

// need to check the file end with .txt
vector<int32_t> getevalindices(const string& result_dir) {
    DIR* dir;
    dirent* entity;
    dir = opendir(result_dir.c_str());
    if (dir) {
        while (entity = readdir(dir)) {
            string path(entity->d_name);
            int32_t len = path.size();
            if (len < 10) continue;
            int32_t index = atoi(path.substr(len - 10, 10).c_str());
            indices.push_back(index);
        }
    }
    return indices;
}

bool eval(string gt_dir, string result_dir, double given_threshold = 0.0){
    // set some global parameters
    initglobals();

    string plot_dir       = result_dir + "/plot";

    // create output directories
    system(("mkdir " + plot_dir).c_str());

    // hold detections and ground truth in memory
    vector< vector<tgroundtruth> > groundtruth;
    vector< vector<tdetection> >   detections;


    vector<bool> eval_ground(num_class, false);

    // for all images read groundtruth and detections
    std::vector<int32_t> indices = getevalindices(result_dir + "/data/");
    //printf("number of files for evaluation: %d\n", (int)indices.size());
    std::cout << "the size of indices is " << indices.size() << std::endl;


    for (int32_t i=0; i<indices.size(); i++) {
        // file name
        char file_name[256];
        sprintf(file_name,"%06d.txt",indices.at(i));
        // read ground truth and result poses
        bool gt_success,det_success;
        vector<tgroundtruth> gt   = loadgroundtruth(gt_dir + "/" + file_name,gt_success);
        vector<tdetection>   det  = loaddetections(result_dir + "/data/" + file_name,
                eval_ground, det_success);
  
        groundtruth.push_back(gt);
        detections.push_back(det);

        // check for errors
        if (!gt_success) {
            cout << "error: couldn't read: %s of ground truth. please write me an email!" << endl;
            return false;
        }
        if (!det_success) {
            cout << "error: couldn't read: %s" << endl;
            return false;
        }
    }
    cout << "  Loading finished.\n";


    // holds pointers for result files
    FILE *fp_det=0, *fp_ori=0;


    if (abs(given_threshold - 0.0) > 0.0001) {
        // eval bird's eye view bounding boxes
        for (int c = 0; c < num_class; c++) {
            classes cls = (classes)c;
            if (eval_ground[c]) {
                if( !eval_class_given_threshold(cls, groundtruth, detections, groundboxoverlap, given_threshold)) {
                    cout << "given threshold evaluation failed." << endl;
                    return false;
                }
            }
        }
    
    } else {
        // eval bird's eye view bounding boxes
        for (int c = 0; c < num_class; c++) {
            classes cls = (classes)c;
            if (eval_ground[c]) {
                fp_det = fopen((result_dir + "/stats_" + class_names[c] + "_detection_ground.txt").c_str(), "w");
                vector<double> precisions;
                vector<double> thresholds;
                vector<double> recalls;
                //if( !eval_class(fp_det, fp_ori, cls, groundtruth, detections, groundboxoverlap, precisions)) {
                if( !new_eval_class(fp_det, fp_ori, cls, groundtruth, detections, groundboxoverlap, recalls, thresholds, precisions)) {
                    cout << " evaluation failed." << endl;
                    return false;
                }
                cout << "I finished new_eval_class" << endl;
                fclose(fp_det);
                //saveandplotplots(plot_dir, class_names[c] + "_detection_bev",class_names[c], precisions);
                newSaveAndPlotPlots(plot_dir, class_names[c] + "_detection_bev", class_names[c], recalls, precisions, thresholds);
            }
        }
    }
    // success
    return true;
}

/*
Labeled ground-truth format is exactly the same with KITTI, we just used fake camera coordinates.
For detection format, we can:
1, convert to fake camera coordinates system, and there is almost nothing to be modified. 
   Note to add the last item as confidence score.
*/

int32_t main (int32_t argc,char *argv[]) {

    if (argc < 3) {
        cout << "Usage: ./eval_detection_bev gt_dir result_dir\n";
        return 1;
    }

    // read arguments
    string gt_dir = argv[1];
    string result_dir = argv[2];
    double given_threshold = 0.0;
    if (argc == 4)
        given_threshold = atof(argv[3]);
    if (argc > 4) {
        cout << "Too many arguments...\n";
        return 1;
    }
    
    // run evaluation
    if (eval(gt_dir, result_dir, given_threshold)) {
        cout << "your evaluation results are available at: " << result_dir.c_str() << endl;
    } else {
        system(("rm -r " + result_dir + "/plot").c_str());
        cout << "an error occured while processing your results.\n";
    }
    return 0;
}
