
#include "GPU_Face_Detect.cuh"
#include <stdio.h>
#include "cuda.h"
#include "lock.h"

//======================================================================================================
// Declare GPU face detection engine variables
//=====================================================================================================

// CUDA performance timers
cudaEvent_t start, stop;

// Device memory haar cascade
GPUHaarCascade h_gpuHaarCascade;
GPUHaarCascade dev_gpuHaarCascade;

// Declare pointers for GPU texture memory
cudaArray * dev_sumArray = NULL;
cudaArray * dev_sqSumArray = NULL;

// Arrays for copying detected faces results from GPU to be post-processed by CPU
GPURect *detectedFaces, *dev_detectedFaces;
size_t detectedFacesSize;


// Initalize device memory for GPU face detection processing
void initGPU(GPUHaarCascade &gpuCascade, IplImage * image, CvMat *sumImg, CvMat *sqSumImg)
{
	int width = image->width;
	int height = image->height;

	//======================================================================================================
	// Define & Init CUDA even timing to determine performance
	//=====================================================================================================

	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );

	//======================================================================================================
	// Define GPU Haar Cascade structures & convert CvHaarCascade to them
	//=====================================================================================================
	
	// Load gpu haar cascade into host copy
	h_gpuHaarCascade.load(&gpuCascade);

	// Allocate device memory
	allocateGPUCascade( h_gpuHaarCascade, dev_gpuHaarCascade);

	//==================================================================
	// Generate integral images & copy them to device texture memory
	//==================================================================

	// Convert double precision sqsum image to float precision
	cv::Mat sqSumMat(sqSumImg->rows, sqSumImg->cols, CV_64FC1, sqSumImg->data.fl);
	//cv::Mat sqSumMat(sqSumImg);
	sqSumMat.convertTo(sqSumMat, CV_32FC1);
	CvMat float_sqsm = sqSumMat;
	
	// Allocate texture memory for integral images & copy host results from OpenCV(cvIntegral()) to device
	allocateIntegralImagesGPU(sumImg, &float_sqsm, dev_sumArray, dev_sqSumArray);

	//===============================================================================
	// Allocate & copy face array data to device memory for storing detection results
	//==============================================================================

	// Allocate memory on the CPU
	detectedFacesSize = width * height * sizeof(GPURect);
	detectedFaces = (GPURect *)malloc(detectedFacesSize);
	memset(detectedFaces, 0, detectedFacesSize);

	// Allocate memory on the GPU & copy host data
	HANDLE_ERROR( cudaMalloc( (void**)&dev_detectedFaces, detectedFacesSize ) );
	HANDLE_ERROR( cudaMemcpy(dev_detectedFaces, detectedFaces, detectedFacesSize, cudaMemcpyHostToDevice));
}

// From array gpuFaces, check each CvRect.width to determine if the GPU determined this window as a valid face
int selectFaces(std::vector<CvRect> &faces, GPURect *gpuFaces, int pixels)
{
	int faces_detected = 0;
	for( int i = 0; i < pixels; i++ )
	{
		// extract the detected rectanlges only 
		GPURect face_rect = gpuFaces[i];
		//CvRect face_rect = gpuFaces[i];
		//printf("Face rect values %f",face_rect.x); 

		if(face_rect.width != 0)
		{

			CvRect r = cvRect((int)face_rect.x,(int) face_rect.y, (int)face_rect.width, (int)face_rect.height);
//CvRect cvRect1(x,y,w,h);
			faces.push_back(r);
			faces_detected++;
		}
	}

	return faces_detected;
}

void startCUDA_EventTming()
{
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );
}

float stopCUDA_EventTiming()
{
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );

	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );

	return elapsedTime;
}

//===============================================================================
// Run v1 kernel for parrelized face detection
//==============================================================================


std::vector<CvRect> runGPUHaarDetection(std::vector<double> scale, int minNeighbors, FaceDetectionKernel kernelSelect)
{
	printf("Beginning GPU(Kernel %d) Haar Detection hello ****\n\n", kernelSelect + 1);
	std::vector<CvRect> faces;
	float totalElapsedTime = 0.0f;
	printf("Scales size:%d",scale.size());
	for(int i = 0; i < scale.size(); i++)
	{
		// Modify all the features for the new scale
		h_gpuHaarCascade.setFeaturesForScale(scale[i]);
		
		// Copy new scaled values over to device
		size_t GPU_Classifier_Size = h_gpuHaarCascade.totalNumOfClassifiers * sizeof(GPUHaarClassifier);
		HANDLE_ERROR( cudaMemcpy(dev_gpuHaarCascade.scaled_haar_classifiers, h_gpuHaarCascade.scaled_haar_classifiers, GPU_Classifier_Size, cudaMemcpyHostToDevice));

		dev_gpuHaarCascade.scale = h_gpuHaarCascade.scale;
		dev_gpuHaarCascade.real_window_size = h_gpuHaarCascade.real_window_size;
		dev_gpuHaarCascade.img_detection_size = h_gpuHaarCascade.img_detection_size;

		int w = dev_gpuHaarCascade.img_detection_size.width;
		int h = dev_gpuHaarCascade.img_detection_size.height;

		// Based on input selection, launch appropriate kernel
		float elapsedTime;
		//printf("Outside case\n");
		switch(kernelSelect)
		{
			case V1:
				printf("Entering launch kernel 1\n");
				elapsedTime = launchKernel_v1(w,  h);
				break;
			/*case V2:
			        printf("entering Launch kernel 2\n");
				elapsedTime = launchKernel_v2(w,  h);
				break;
			case V3:
					printf("entering Launch	kernel 3\n");			
					elapsedTime = launchKernel_v3(w, h);
				break;
			case V4:
					printf("entering Launch	kernel 4\n");
				elapsedTime = launchKernel_v4(w,  h);
				break;*/
		}

		totalElapsedTime += elapsedTime;

		// Copy results from device & process them on CPU
		HANDLE_ERROR( cudaMemcpy(detectedFaces, dev_detectedFaces, detectedFacesSize, cudaMemcpyDeviceToHost));
	
		// Scan detectedFaces array from GPU for valid detected faces
		int faces_detected = selectFaces(faces, detectedFaces, w * h);

		// Output performance information for this stage
		printf("Stage: %d // Faces Detected: %d // GPU Time: %3.1f ms \n", i, faces_detected, elapsedTime);
	}

	// Output final performance
	printf("\nTotal compute time: %3.1f ms \n\n", totalElapsedTime);

	// Group detected faces for cleaner results
	if( minNeighbors != 0)
	{
		groupRectangles(faces, minNeighbors, GROUP_EPS);
	}

	// Clean up detected faces arrays for future processing
	memset(detectedFaces, 0, detectedFacesSize);
	HANDLE_ERROR( cudaMemcpy(dev_detectedFaces, detectedFaces, detectedFacesSize, cudaMemcpyHostToDevice));

	return faces;
}

float launchKernel_v1(int width, int height)
{
	// Define number of blocks and threads to divide work
	dim3    blocks(width/16, height/16);
	dim3    threads(16, 16);

	// Begin CUDA timing performance
	startCUDA_EventTming();

		// Call kerenel on GPU to run haarcascade for every detection window in image
		haarDetection_v1<<<blocks, threads>>>(dev_gpuHaarCascade, dev_detectedFaces);

	// Stop CUDA timing performance
	return stopCUDA_EventTiming(); 
}

void shutDownGPU()
{
	releaseTextures();

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );

	h_gpuHaarCascade.shutdown();
	HANDLE_ERROR( cudaFree(dev_gpuHaarCascade.haar_classifiers));
	HANDLE_ERROR( cudaFree(dev_gpuHaarCascade.scaled_haar_classifiers));

	free(detectedFaces);
	HANDLE_ERROR( cudaFree(dev_detectedFaces));

	HANDLE_ERROR( cudaFreeArray(dev_sumArray));
	HANDLE_ERROR( cudaFreeArray(dev_sqSumArray));

	HANDLE_ERROR( cudaDeviceReset());
}

