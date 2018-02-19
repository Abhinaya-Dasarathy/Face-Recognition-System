
#include "GPU_Face_Detect.cuh"

#include "cuda.h"

#include "lock.h"

#define CONSTANT_MEM_SIZE 32
__constant__ GPUHaarStageClassifier stageClassifiers[CONSTANT_MEM_SIZE];

texture<int, 2, cudaReadModeElementType> sumImageRef;
texture<float, 2, cudaReadModeElementType> sqSumImageRef;

void allocateGPUCascade(GPUHaarCascade &h_gpuCascade, GPUHaarCascade &dev_gpuCascade)
{
	// copy generic parameters
	dev_gpuCascade.flags = h_gpuCascade.flags;
	dev_gpuCascade.numOfStages = h_gpuCascade.numOfStages;
	dev_gpuCascade.orig_window_size = h_gpuCascade.orig_window_size;
	dev_gpuCascade.real_window_size = h_gpuCascade.real_window_size;
	dev_gpuCascade.img_window_size = h_gpuCascade.img_window_size;
	dev_gpuCascade.scale = h_gpuCascade.scale;
	dev_gpuCascade.totalNumOfClassifiers = h_gpuCascade.totalNumOfClassifiers;

	// Allocate space for device classifiers and copy classifiers from host
	size_t GPU_Classifier_Size = h_gpuCascade.totalNumOfClassifiers * sizeof(GPUHaarClassifier);
	HANDLE_ERROR( cudaMalloc( (void**)&dev_gpuCascade.haar_classifiers, GPU_Classifier_Size ) );
	HANDLE_ERROR( cudaMemcpy(dev_gpuCascade.haar_classifiers, h_gpuCascade.haar_classifiers, GPU_Classifier_Size, cudaMemcpyHostToDevice));

	HANDLE_ERROR( cudaMalloc( (void**)&dev_gpuCascade.scaled_haar_classifiers, GPU_Classifier_Size ) );
	HANDLE_ERROR( cudaMemcpy(dev_gpuCascade.scaled_haar_classifiers, h_gpuCascade.scaled_haar_classifiers, GPU_Classifier_Size, cudaMemcpyHostToDevice));


	if(h_gpuCascade.numOfStages > CONSTANT_MEM_SIZE)
	{
		printf("ERROR: Number of stages is larger than the max size of constant memory alloted");
		system("pause");
		return;
	}

	HANDLE_ERROR( cudaMemcpyToSymbol( stageClassifiers, h_gpuCascade.haar_stage_classifiers, sizeof(GPUHaarStageClassifier) * h_gpuCascade.numOfStages ) );
}

void allocateIntegralImagesGPU(CvMat * sumImage, CvMat *sqSumImage, cudaArray *dev_sumArray, cudaArray * dev_sqSumArray)
{
	//===========================================================
	// Allocate & reference texture memory for sum integral image
	//===========================================================

	// Create channel descripition for texture( 1 channel 32 bits, type signed int)
	cudaChannelFormatDesc sum_channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned );

	// Allocate device memory for sum image texture
	HANDLE_ERROR( cudaMallocArray(&dev_sumArray, &sum_channelDesc, sumImage->width, sumImage->height));

	// Copy image data from OpenCv to device memory
	HANDLE_ERROR(cudaMemcpy2DToArray(dev_sumArray, 0, 0, sumImage->data.i, sumImage->step, sumImage->width * sizeof(int), sumImage->height, cudaMemcpyHostToDevice));
	
	// Set parameters for CUDA texture reference
	sumImageRef.addressMode[0] = cudaAddressModeWrap;
	sumImageRef.addressMode[1] = cudaAddressModeWrap;
	sumImageRef.filterMode = cudaFilterModePoint; //cudaFilterModeLinear
	sumImageRef.normalized = false;

	// Bind texture reference to our allocated device memory	
	HANDLE_ERROR( cudaBindTextureToArray(sumImageRef, dev_sumArray, sum_channelDesc));

	//==================================================================
	// Allocate & reference texture memory for square sum integral image
	//==================================================================

	// Create channel descripition for texture( 1 channel 64 bits, type float)
	cudaChannelFormatDesc sqSum_channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	
	// Allocate device memory for sum image texture
	HANDLE_ERROR( cudaMallocArray(&dev_sqSumArray, &sqSum_channelDesc, sqSumImage->width, sqSumImage->height));

	// Copy image data from OpenCv to device memory
	HANDLE_ERROR(cudaMemcpy2DToArray(dev_sqSumArray, 0, 0, sqSumImage->data.fl, sqSumImage->step, sqSumImage->width * sizeof(int), sqSumImage->height, cudaMemcpyHostToDevice));
	
	// Set parameters for CUDA texture reference
	sqSumImageRef.addressMode[0] = cudaAddressModeWrap;
	sqSumImageRef.addressMode[1] = cudaAddressModeWrap;
	sqSumImageRef.filterMode = cudaFilterModeLinear;
	sqSumImageRef.normalized = false;

	// Bind texture reference to our allocated device memory
	HANDLE_ERROR( cudaBindTextureToArray(sqSumImageRef, dev_sqSumArray, sqSum_channelDesc));
}

void releaseTextures()
{
	cudaUnbindTexture(sumImageRef);
	cudaUnbindTexture(sqSumImageRef);
}

//===============================================================================
//
//
//===============================================================================

__device__ float calculateMean(GPURect rect)
{
	int A = tex2D(sumImageRef, rect.x, rect.y);
	int B = tex2D(sumImageRef, rect.x + rect.width, rect.y);
	int C = tex2D(sumImageRef, rect.x + rect.width, rect.y + rect.height);
	int D = tex2D(sumImageRef, rect.x, rect.y + rect.height);

	return (float)(A - B + C - D);
}


__device__ float calculateSum(GPURect rect, int win_start_x, int win_start_y)
{
	float tx = win_start_x + rect.x;
	float ty = win_start_y + rect.y;

	int A = tex2D(sumImageRef, tx, ty);
	int B = tex2D(sumImageRef, tx + rect.width, ty);
	int C = tex2D(sumImageRef, tx + rect.width, ty + rect.height);
	int D = tex2D(sumImageRef, tx, ty + rect.height);

	return (float)(A - B + C - D);
}

__device__ int getOffset(int x, int y)
{
	// blockDim.x * gridDim.x; = img.width
	return x + y * blockDim.x * gridDim.x;
}

__device__ float runHaarFeature(GPUHaarClassifier classifier, GPURect detectionWindow, float variance_norm_factor, float weightScale)
{
	double t = classifier.threshold * variance_norm_factor;

	double sum = calculateSum(classifier.haar_feature.rect0.r, detectionWindow.x, detectionWindow.y) * classifier.haar_feature.rect0.weight * weightScale;
	sum += calculateSum(classifier.haar_feature.rect1.r, detectionWindow.x, detectionWindow.y) * classifier.haar_feature.rect1.weight * weightScale;

	// If there is a third rect
	if(classifier.haar_feature.rect2.weight)
		sum += calculateSum(classifier.haar_feature.rect2.r, detectionWindow.x, detectionWindow.y) * classifier.haar_feature.rect2.weight * weightScale;
            
	if(sum >= t)
		return classifier.alpha1;
	else
		return classifier.alpha0;
}

__device__ float calculateVariance(GPURect detectionWindow)
{
	float inv_window_area = 1.0f / ((float)detectionWindow.width * detectionWindow.height);
	float weightScale = inv_window_area;

	// HaarCascade file requires normalization of features
	float mean = calculateMean(detectionWindow) * inv_window_area;
	
	float variance_norm_factor = tex2D(sqSumImageRef, detectionWindow.x, detectionWindow.y) - 
		tex2D(sqSumImageRef, detectionWindow.x + detectionWindow.width, detectionWindow.y) - 
		tex2D(sqSumImageRef, detectionWindow.x, detectionWindow.y + detectionWindow.height) + 
		tex2D(sqSumImageRef, detectionWindow.x + detectionWindow.width, detectionWindow.y + detectionWindow.height);
		
	variance_norm_factor = variance_norm_factor * inv_window_area - mean * mean;
	//variance_norm_factor = sqrt(variance_norm_factor * inv_window_area - mean * mean);

	if(variance_norm_factor >= 0.0f)
		variance_norm_factor = sqrt(variance_norm_factor);
	else
	{		
		variance_norm_factor = 1.0f;
	}

	return variance_norm_factor;
}



__global__ void haarDetection_v1(GPUHaarCascade haarCascade, GPURect * detectedFaces)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = getOffset(x,y);

	// If current pixel is out of bounds, just return
	
	if(x < haarCascade.img_detection_size.width || y < haarCascade.img_detection_size.height)
	{
		GPURect detectionWindow;
		detectionWindow.x = x;
		detectionWindow.y = y;
		detectionWindow.width = haarCascade.real_window_size.width;
		detectionWindow.height = haarCascade.real_window_size.height;

		float variance_norm_factor = calculateVariance(detectionWindow);
		float inv_window_area = 1.0f / ((float)detectionWindow.width * detectionWindow.height);
		float weightScale = inv_window_area;

		// Assume face was detected
		detectedFaces[offset] = detectionWindow;

		//bool faceDetected = true;
		// for each stage in cascade
		for(int i = 0; i < haarCascade.numOfStages; i++)
		{
			float stage_sum = 0.0;
			for(int j = 0; j < stageClassifiers[i].numofClassifiers; j++)
			{
				int index = j + stageClassifiers[i].classifierOffset;
				GPUHaarClassifier classifier = haarCascade.scaled_haar_classifiers[index];
				//GPUHaarClassifier classifier = haarCascade.haar_classifiers[index];

				stage_sum += runHaarFeature(classifier, detectionWindow, variance_norm_factor, inv_window_area);
			}

			// Classifier did not pass, abort entire cascade
			if( stage_sum < stageClassifiers[i].threshold)
			{
				// Set width to zero to indicate on CPU side that this is not a face
				detectedFaces[offset].width = 0;
				break;
			}
		}
	}
}

