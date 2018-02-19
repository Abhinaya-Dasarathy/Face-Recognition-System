/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


using namespace cv;
using namespace std;

 #define MAX_IMAGE 200000


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {

  std::ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    string error_message = "No valid input file was given, please check the given filename.";
    CV_Error(CV_StsBadArg, error_message);
  }

  string line, path, classlabel;
  int position;
  Mat A, B;
  char outbuf[MAX_IMAGE];
  unsigned char *data;
  int rows, cols, label, i;
  while (getline(file, line)) {
    stringstream liness(line);

    getline(liness, path, separator);
    
    getline(liness, classlabel);

    if(!path.empty() && !classlabel.empty()) {
      position = 0;
      label = atoi(classlabel.c_str());
      A = imread(path, 0);
      B = Mat(A.rows, A.cols, 0, A.data);
      imwrite(format("c0/testemean%i.png", i++), B);
      images.push_back(A);
      labels.push_back(label);

    }
  }
}

int main(int argc, char *argv[]) {
  // Check for valid command line arguments, print usage
  // if no arguments were given.
  
  if (argc < 2) {
    cout << "usage: " << argv[0] << " <csv.ext> <input image>" << endl;
    exit(1);
  }
    
  int local_n;


  cout<<"Initiating Process "<<endl;
  // Get the path to your CSV.
  string fn_csv = string(argv[1]);
   
  // These vectors hold the images and corresponding labels.
  vector<Mat> images;
  vector<int> labels;

  // Read in the data. This can fail if no valid
  // input filename is given.
  try {
    read_csv(fn_csv, images, labels);
  } catch (cv::Exception& e) {
    cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
    // nothing more we can do
    exit(1);
  }

  // Quit if there are not enough images for this demo.
  if(images.size() <= 1) {
    string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
    CV_Error(CV_StsError, error_message);
  }
    
  int height = images[0].rows;

  Mat testSample;
  int position, rows, cols, i, j;
  double start, finish, elapsed, local_elapsed;

  char inputbuf[MAX_IMAGE];
  unsigned char *inputdata;
  int predictedLabel;

  int predictImage;
  double local_confidence = 0.0;
  double confidence = 0.0;
    
  position = 0;


  // string test = argv[2];
  // string classlabel;
  // stringstream testlabels(test);
  // getline(testlabels, classlabel,'/');
  // getline(testlabels , classlabel,'/');
  // string sublabel = classlabel.substr(1,classlabel.find('/')-1);
  // int label;
  // label = atoi(sublabel.c_str());

  //reading desired image


  testSample = imread(argv[2], 0);
       

  cout<<"Process is buiding a model."<<endl;
  Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
  model->train(images, labels);

  predictedLabel = model->predict(testSample);
  cout<<"predicted label of Process is "<<predictedLabel<<endl;
  predictImage = predictedLabel;
  cout<<"display_results"<<endl;
  string result_message = format("Predicted class = %d ", predictImage);

        
       
  IplImage* n[10];
  cout<<"Creating Output Face which is recognized"<<endl;
  IplImage* dst=cvCreateImage(cvSize(5*images[0].cols,2*images[0].rows),IPL_DEPTH_8U,3);
  for(i = 0; i < 5; i++){
    for(j = 0; j < 2; j++){
      n[(i + 4*j)] = cvLoadImage(format("./database/s%i/%i.pgm", predictImage, (i + 4*j)+1).c_str());
      cvSetImageROI(dst, cvRect(i*n[(i + 4*j)]->width, j*n[(i + 4*j)]->height,n[(i + 4*j)]->width,n[(i + 4*j)]->height) );
      cvCopy(n[(i + 4*j)],dst,NULL);
      cvResetImageROI(dst);
    }
  }
  int compression_params[3];
  compression_params[0] = 1;
  compression_params[1] = 10;
  compression_params[2] = 0;
  cout<<"Writing the output to RecognizedFaceSerial.pgm"<<endl;
  const char *file_name = "./RecognizedFaceSerial.pgm";
  cvSaveImage(file_name, dst, compression_params);
  return 0;
}
