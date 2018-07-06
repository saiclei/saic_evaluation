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


/*=======================================================================
static evaluation parameters
=======================================================================*/

// evaluated object classes
enum classes{car=0, pedestrian=1, truck=3};
const int num_class = 4;

// parameters varying per class
vector<string> class_names;
// the minimum overlap required for 2d evaluation on the image/ground plane and 3d evaluation7const double min_overlap[3] = {0.5, 0.5, 0.5};7
// // no. of recall steps that should be evaluated (discretized)
const double min_overlap[3] = {0.5, 0.2, 0.5};

const double n_sample_pts = 41;
const int n_recall_step = 4;
const double d_sum_denominator = 11;

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
  int32_t        tp;          // true positives
  int32_t        fp;          // false positives
  int32_t        fn;          // false negatives
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

vector<double> getthresholds(vector<double> &v, double n_groundtruth){

  // holds scores needed to compute n_sample_pts recall values
  vector<double> t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for(int32_t i=0; i<v.size(); i++){

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i+1)/n_groundtruth;
    if(i<(v.size()-1))
      r_recall = (double)(i+2)/n_groundtruth;
    else
      r_recall = l_recall;

    if( (r_recall-current_recall) < (current_recall-l_recall) && i<(v.size()-1))
      continue;

    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0/(n_sample_pts-1.0);
  }
  return t;
}

void cleandata(classes current_class, const vector<tgroundtruth> &gt, const vector<tdetection> &det, vector<int32_t> &ignored_gt, vector<int32_t> &ignored_det, int32_t &n_gt){
  // extract ground truth bounding boxes for current evaluation class
    for (int i = 0; i < gt.size(); ++i) {
        if(!strcasecmp(gt[i].box.type.c_str(), class_names[current_class].c_str())) {
            ignored_gt.push_back(0);
            n_gt++;
        } else {
            ignored_gt.push_back(-1);
        }
    }

    // extract detections bounding boxes of the current class
    for(int32_t i=0;i<det.size(); i++){
        if(!strcasecmp(det[i].box.type.c_str(), class_names[current_class].c_str())) {
            ignored_det.push_back(0);
        } else {
            ignored_det.push_back(-1);
        }
    }

}

tprdata computestatistics(classes current_class, const vector<tgroundtruth> &gt,
        const vector<tdetection> &det, const vector<int32_t> &ignored_gt, const vector<int32_t>  &ignored_det,
        bool compute_fp, double (*boxoverlap)(tdetection, tgroundtruth, int32_t),
        double thresh=0, bool debug=false){

  tprdata stat = tprdata();
  const double no_detection = -10000000;
  vector<bool> assigned_detection; // holds wether a detection was assigned to a valid or ignored ground truth
  assigned_detection.assign(det.size(), false);
  vector<bool> ignored_threshold;
  ignored_threshold.assign(det.size(), false); // holds detections with a threshold lower than thresh if fp are computed

  // detections with a low score are ignored for computing precision (needs fp)
  if(compute_fp)
    for(int32_t i=0; i<det.size(); i++)
      if(det[i].thresh<thresh)
        ignored_threshold[i] = true;

  // evaluate all ground truth boxes
  for(int32_t i=0; i<gt.size(); i++){

    // this ground truth is not of the current or a neighboring class and therefore ignored
    if(ignored_gt[i]==-1)
      continue;

    /*=======================================================================
    find candidates (overlap with ground truth > 0.5) (logical len(det))
    =======================================================================*/
    int32_t det_idx          = -1;
    double valid_detection = no_detection;
    double max_overlap     = 0;

    // search for a possible detection
    bool assigned_ignored_det = false;
    for(int32_t j=0; j<det.size(); j++){

      // detections not of the current class, already assigned or with a low threshold are ignored
      if(ignored_det[j]==-1)
        continue;
      if(assigned_detection[j])
        continue;
      if(ignored_threshold[j])
        continue;

      // find the maximum score for the candidates and get idx of respective detection
      double overlap = boxoverlap(det[j], gt[i], -1);

      // for computing recall thresholds, the candidate with highest score is considered
      if(!compute_fp && overlap>min_overlap[current_class] && det[j].thresh>valid_detection){
        det_idx         = j;
        valid_detection = det[j].thresh;
      }

      // for computing pr curve values, the candidate with the greatest overlap is considered
      else if(compute_fp && overlap>min_overlap[current_class] && (overlap>max_overlap || assigned_ignored_det) && ignored_det[j]==0){
        max_overlap     = overlap;
        det_idx         = j;
        valid_detection = 1;
        assigned_ignored_det = false;
      }
      else if(compute_fp && overlap>min_overlap[current_class] && valid_detection==no_detection && ignored_det[j]==1){
        det_idx              = j;
        valid_detection      = 1;
        assigned_ignored_det = true;
      }
    }

    /*=======================================================================
    compute tp, fp and fn
    =======================================================================*/

    // nothing was assigned to this valid ground truth
    if(valid_detection==no_detection && ignored_gt[i]==0) {
      stat.fn++;
    }

    // only evaluate valid ground truth <=> detection assignments (considering difficulty level)
    else if(valid_detection!=no_detection && (ignored_gt[i]==1 || ignored_det[det_idx]==1))
      assigned_detection[det_idx] = true;

    // found a valid true positive
    else if(valid_detection!=no_detection){

      // write highest score to threshold vector
      stat.tp++;
      stat.v.push_back(det[det_idx].thresh);


      // clean up
      assigned_detection[det_idx] = true;
    }
  }

  // if fp are requested, consider stuff area
  if(compute_fp) {

    // count fp
    for (int32_t i = 0; i < det.size(); i++) {

      // count false positives if required (height smaller than required is ignored (ignored_det==1)
      if (!(assigned_detection[i] || ignored_det[i] == -1 || ignored_det[i] == 1 || ignored_threshold[i]))
        stat.fp++;
    }

  }
  return stat;
}

/*=======================================================================
evaluate class-wise
=======================================================================*/

bool eval_class (FILE *fp_det, FILE *fp_ori,classes current_class,
                 const vector< vector<tgroundtruth> > &groundtruth,
                 const vector< vector<tdetection> > &detections,
                  double (*boxoverlap)(tdetection, tgroundtruth, int32_t),
                 vector<double> &precision) {

	assert(groundtruth.size() == detections.size());
  	// init
  	int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
  	vector<double> v, thresholds;                       // detection scores, evaluated for recall discretization
  	vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/difficulty

  	// for all test images do
  	for (int32_t i=0; i<groundtruth.size(); i++){
    	// holds ignored ground truth, ignored detections and dontcare areas for current frame
    	vector<int32_t> i_gt, i_det;

    	// only evaluate objects of current class and ignore occluded, truncated objects
    	cleandata(current_class, groundtruth[i], detections[i], i_gt, i_det, n_gt);
    	ignored_gt.push_back(i_gt);
    	ignored_det.push_back(i_det);

    	// compute statistics to get recall values
    	tprdata pr_tmp = tprdata();
    	pr_tmp = computestatistics(current_class, groundtruth[i], detections[i], i_gt, i_det, false, boxoverlap, false, false);

    	// add detection scores to vector over all images
    	for(int32_t j=0; j<pr_tmp.v.size(); j++)
      	v.push_back(pr_tmp.v[j]);
  	}

  	// get scores that must be evaluated for recall discretization
  	thresholds = getthresholds(v, n_gt);
    // thresholds = getEachThresholds(v);
  	// compute tp,fp,fn for relevant scores
  	vector<tprdata> pr;
  	pr.assign(thresholds.size(),tprdata());
  	for (int32_t i=0; i<groundtruth.size(); i++){

    	// for all scores/recall thresholds do:
    	for(int32_t t=0; t<thresholds.size(); t++){
      		tprdata tmp = tprdata();
      		tmp = computestatistics(current_class, groundtruth[i], detections[i], 
                              ignored_gt[i], ignored_det[i], true, boxoverlap,
                              thresholds[t], t==38);

      		// add no. of tp, fp, fn, aos for current frame to total evaluation for current threshold
      		pr[t].tp += tmp.tp;
      		pr[t].fp += tmp.fp;
      		pr[t].fn += tmp.fn;
    	}
  	}

  	// compute recall, precision and aos
  	vector<double> recall;
  	precision.assign(n_sample_pts, 0);

  	double r=0;
  	for (int32_t i=0; i<thresholds.size(); i++){
    	r = pr[i].tp/(double)(pr[i].tp + pr[i].fn);
    	recall.push_back(r);
    	precision[i] = pr[i].tp/(double)(pr[i].tp + pr[i].fp);

  	}

  	// filter precision and aos using max_{i..end}(precision)
  	for (int32_t i=0; i<thresholds.size(); i++){
    	precision[i] = *max_element(precision.begin()+i, precision.end());
  	}

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
    	cleandata(current_class, groundtruth[i], detections[i], i_gt, i_det, n_gt);
    	ignored_gt.push_back(i_gt);
    	ignored_det.push_back(i_det);

    	// compute statistics to get recall values
    	tprdata pr_tmp = tprdata();

  	}
    
    double tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < groundtruth.size(); ++i) {
        tprdata tmp = tprdata();
        tmp = computestatistics(current_class, groundtruth[i], detections[i], 
                              ignored_gt[i], ignored_det[i], true, boxoverlap,
                              given_threshold);
        tp += tmp.tp;
        fp += tmp.fp;
        fn += tmp.fn;
    }
    cout << "tp, fp and fn are: " << tp << ", " << fp << ", and " << fn << endl;
  	double recall = tp / (tp + fn);
    double precision = tp / (tp + fp);
    cout << "Given confidence socre thresold is: " << given_threshold << ", recall is: " 
         << recall << " and precision is: " << precision << endl;

    FILE *fp_curve = fopen(("/home/saiclei/curve_" + class_names[current_class] + ".txt").c_str(),"a+");
    fprintf(fp_curve, "%f %f\n", precision, recall);
    fclose(fp_curve);
   return true; 
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
                vector<double> precision;
                if( !eval_class(fp_det, fp_ori, cls, groundtruth, detections, groundboxoverlap, precision)) {
                    cout << " evaluation failed." << endl;
                    return false;
                }
                fclose(fp_det);
                saveandplotplots(plot_dir, class_names[c] + "_detection_bev",class_names[c], precision);
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
