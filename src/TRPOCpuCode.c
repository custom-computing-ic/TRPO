#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"


typedef struct {
        // Model Parameter File Name - weight, bias, std.
        char * ModelFile;
        // Simulation Data File Name - probability and observation
        char * DataFile;
        // Number of Layers in the network: [Input] --> [Hidden 1] --> [Hidden 2] --> [Output] is 4 layers.
        size_t NumLayers;
        // Activation Function of each layer: t = tanh, l = linear (activate(x)=x), s = sigmoid
        char * AcFunc;
        // Number of Nodes: from [Input] to [Output]
        // InvertedPendulum-v1: 4, 64, 64, 1
        // Humanoid-v1: 367, 64, 64, 17        
        size_t * LayerSize;
        // Number of Samples
        size_t NumSamples;
        
 
        
} TRPOparam;


int FVP (TRPOparam param, double *result, double *input) 
{

        //////////////////// Arguments ////////////////////
        // param: TRPO parameters
        // result: the Fisher-Vector Product
        // input: the vector to be multiplied with the Fisher Information Matrix
        // observ: list of observations - corresponds to ob_no in modular_rl
        // prob: list of probablity mean and std values - corresponds to prob_np in modular_rl
        // W: Weights in the neural network         double * W [NumLayers-1];

        // Assign Parameters
        const size_t NumLayers = param.NumLayers;
        char * AcFunc = param.AcFunc;
        size_t * LayerSize = param.LayerSize;
        const size_t NumSamples = param.NumSamples;
        char * ModelFile = param.ModelFile;
        char * DataFile = param.DataFile;
        
        // Dimension of Observation Space
        const size_t ObservSpaceDim = LayerSize[0];
        // Dimension of Action Space
        const size_t ActionSpaceDim = LayerSize[NumLayers-1];

        
        //////////////////// Memory Allocation ////////////////////

        // Allocate Memory for each layer and the Gradient w.r.t. the pre-activation values in each layer
        // Remarks: This gradient storage is optional, as we can probably overwrite the values in Layer[i]
        //                  during the backpropagation process. Here we store it explicitly for easy debugging.
        //                  The space complexity is just O(Number of Neurons), which is a very small number
        //                  campared to the number of weights. So it's fine.
        double * Layer [NumLayers];
        double * GLayer [NumLayers];
        for (size_t i=0; i<NumLayers; ++i) {
                Layer[i] = (double *) calloc(LayerSize[i], sizeof(double));
                GLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
        }
        
        // Allocate Memory for Weight Matrix W and Bias Vector B
        // W[i] and B[i] is the Weight Matrix and Bias Vector from Layer[i] to Layer[i+1]
        // Item (j,k) in W[i] refers to the weight from Neuron #j in Layer[i] to Neuron #k in Layer[i+1]
        // The Gradient for W[i] and B[i] are dW[i] and dB[i]
        // Remarks: B[i]
        double * W [NumLayers-1];
        double * dW [NumLayers-1];
        double * B [NumLayers-1];
        double * dB [NumLayers-1];
        for (size_t i=0; i<NumLayers-1; ++i) {
                W[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
                dW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
                B[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
                dB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
        }

        // Allocate Memory for Observation
        double * observ = (double *) calloc(NumSamples*ObservSpaceDim, sizeof(double));
        
        // Allocate Memory for Probability Mean
        double * mean = (double *) calloc(NumSamples*ActionSpaceDim, sizeof(double));

        // Allocate Memory for Probability Std
        double * std = (double *) calloc(ActionSpaceDim, sizeof(double));

        
        //////////////////// Parameter Initialisation ////////////////////
        
        // Open Model File that contains Weights, Bias and std
	FILE *ModelFilePointer = fopen(ModelFile, "r");
	if (ModelFilePointer==NULL) {
		fprintf(stderr, "[ERROR] Cannot open Model File [%s]. \n", ModelFile);
  		return -1;
	}        
        
        // Read Weights and Bias from file
        for (size_t i=0; i<NumLayers-1; ++i) {
                // Reading Weights W[i]: from Layer[i] to Layer[i+1]
                size_t curLayerDim = LayerSize[i];
                size_t nextLayerDim = LayerSize[i+1];
                for (size_t j=0; j<curLayerDim;++j) {
                        for (size_t k=0; k<nextLayerDim; ++k) {
                                fscanf(ModelFilePointer, "%lf", &W[i][j*nextLayerDim+k]);
                        }
                }
                // Reading Bias B[i]: from Layer[i] to Layer[i+1]
                for (size_t k=0; k<nextLayerDim; ++k) {
                        fscanf(ModelFilePointer, "%lf", &B[i][k]);
                }
        }

        // Read std from file
        for (size_t k=0; k<ActionSpaceDim; ++k) {
                fscanf(ModelFilePointer, "%lf", &std[k]);
        }

        // Close Model File
        fclose(ModelFilePointer);
        
        
        //////////////////// Read Simulation Data ////////////////////
        
        // Open Data File that contains Mean, std and Observation
	FILE *DataFilePointer = fopen(DataFile, "r");
	if (DataFilePointer==NULL) {
		fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", DataFile);
  		return -1;
	}  
        
        // Read Mean, std and Observation
        for (size_t i=0; i<NumSamples; ++i) {
                // Read Mean
                for (size_t j=0; j<ActionSpaceDim; ++j) {
                        fscanf(DataFilePointer, "%lf", &mean[i*ActionSpaceDim+j]);
                }
                // Read Std
                for (size_t j=0; j<ActionSpaceDim; ++j) {
                        fscanf(DataFilePointer, "%lf", &std[j]);
                }
                // Read Observation
                for (size_t j=0; j<ObservSpaceDim; ++j) {
                        fscanf(DataFilePointer, "%lf", &observ[i*ObservSpaceDim+j]);
                }
        }

        
        //////////////////// Main Loop Over All Samples ////////////////////        
        
        for (size_t iter=0; iter<NumSamples; iter++) {
        
                //////////////////// Forward Propagation ////////////////////
        
                // Assign Input Values
                for (size_t i=0; i<ObservSpaceDim; ++i) Layer[0][i] = observ[iter*ObservSpaceDim+i];
        
                // Forward Propagation
                for (size_t i=0; i<NumLayers-1; ++i) {
                        
                        // Propagate from Layer[i] to Layer[i+1]
                        for (size_t j=0; j<LayerSize[i+1]; ++j) {
                                
                                // Calculating item[j] in next layer
                                Layer[i+1][j] = B[i][j];
//                              printf("Net[%zu][%zu] init to bias %f \n", i+1, j, Layer[i+1][j]);
                                for (size_t k=0; k<LayerSize[i]; ++k) {
                                        // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                                        Layer[i+1][j] += Layer[i][k] * W[i][k*LayerSize[i+1]+j];
//                              printf("Net[%zu][%zu] += Input %f * Weight %f => %f \n", i+1, j, Layer[i][k], W[i][k*LayerSize[i+1]+j], Layer[i+1][j]);
                                }
//                              printf("Net[%zu][%zu] = %f => ", i+1, j, Layer[i+1][j]);
                        
                                 // Apply Activation Function
                                switch (AcFunc[i+1]) {
                                        // Linear Activation Function: Ac(x) = (x)
                                        case 'l': {break;}
                                        // tanh() Activation Function
                                        case 't': {Layer[i+1][j] = tanh(Layer[i+1][j]); break;}
                                        // sigmoid Activation Function
                                        case 's': {Layer[i+1][j] = 1.0/(1+exp(-Layer[i+1][j])); break;}
                                        // Default: Activation Function not supported
                                        default: {
                                                printf("[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                                                return -1;
                                        }
                                }
//                              printf("Out[%zu][%zu] = %f \n", i+1, j, Layer[i+1][j]);
                        }
                }
                
                // Print Final Output
                for (size_t i=0; i<ActionSpaceDim; ++i) {
                        printf("output[%zu] = %f \n", i, Layer[NumLayers-1][i]);
                }

                
                //////////////////// Backward Propagation - First Round ////////////////////                 

        
                // Assign the derivative of KL w.r.t. the output values from the final layer
                for (size_t i=0; i<ActionSpaceDim; ++i) {
//                        GLayer[NumLayers-1][i] = 0.1 * ()
                }
                
                
                
                GLayer[NumLayers-1][0] = Layer[NumLayers-1][0] - 0.01;
                GLayer[NumLayers-1][1] = Layer[NumLayers-1][1] - 0.99;                

                
                // Backward Propagation
                for (size_t i=NumLayers-1; i>0; --i) {
       
                        // Propagate from Layer[i] to Layer[i-1]
                        for (size_t j=0; j<LayerSize[i]; ++j) {

                                // Calculate derivative of the activation function
                                switch (AcFunc[i]) {
                                        // Linear Activation Function: Ac(x) = (x)
                                        case 'l': {break;}
                                        // tanh() Activation Function: tanh' = 1 - tanh^2
                                        case 't': {GLayer[i][j] = GLayer[i][j] * (1- Layer[i][j] * Layer[i][j]); break;}
                                        // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                                        case 's': {GLayer[i][j] = GLayer[i][j] * Layer[i][j] * (1- Layer[i][j]); break;}
                                        // Default: Activation Function not supported
                                        default: {
                                                fprintf(stderr, "[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i+1, AcFunc[i]);
                                                return -1;
                                        }
                                }
                                
                                // The derivative w.r.t to Bias is the same as that w.r.t. the pre-activated value
                                dB[i-1][j] = GLayer[i][j];
                        }
                
                        // Calculate the derivative w.r.t. to Weight
                        for (size_t j=0; j<LayerSize[i-1]; ++j) {
                                for (size_t k=0; k<LayerSize[i]; ++k) {
                                        // The Derivative w.r.t. to the weight from Neuron #j in Layer[i-1] to Neuron #k in Layer[i]
                                        dW[i-1][j*LayerSize[i]+k] = GLayer[i][k] * Layer[i-1][j];
                                        printf("Gradient w.r.t. Weight[%zu][%zu][%zu] = %f \n", i-1, j, k, dW[i-1][j*LayerSize[i]+k]);
                                }
                        }
                
                        // Calculate the derivative w.r.t. the output values from Layer[i]
                        for (size_t j=0; j<LayerSize[i-1]; ++j) {
                                GLayer[i-1][j] = 0;
                                for (size_t k=0; k<LayerSize[i]; ++k) {
                                        // Accumulate the Gradient from Neuron #k in Layer[i] to Neuron #j in Layer[i-1]
                                        GLayer[i-1][j] += GLayer[i][k] * W[i-1][j*LayerSize[i]+k];
                                }
                        }
                
                } // End of Back Propagation
        
        } // End of iteration over current sample


        //////////////////// Clean Up ////////////////////  

        // clean up
        for (size_t i=0; i<NumLayers; ++i) {free(Layer[i]); free(GLayer[i]);}
        for (size_t i=0; i<NumLayers-1; ++i) {free(W[i]); free(dW[i]); free(B[i]); free(dB[i]);}
        free(observ); free(mean); free(std);

        return 0;
}




int main()
{

        //////////////////// C Version of Back Propagation ////////////////////
        
        // Simple Data 2-2-2
        char AcFunc [] = {'l', 's', 's'};
        size_t LayerSize [] = {2, 2, 2};

        TRPOparam SimpleDataParam;
        SimpleDataParam.ModelFile = "SimpleModel2-2-2.txt";
        SimpleDataParam.DataFile = "SimpleModel2-2-2Data.txt";
        SimpleDataParam.NumLayers = 3;
        SimpleDataParam.AcFunc = AcFunc;
        SimpleDataParam.LayerSize = LayerSize;
        SimpleDataParam.NumSamples = 1;

        double dummy_result=0;
        double dummy_input=1;

        int FVPStatus = FVP (SimpleDataParam, &dummy_result, &dummy_input);



        //////////////////// FPGA ////////////////////


        const int inSize = 384;

        int *a = malloc(sizeof(int) * inSize);
        int *b = malloc(sizeof(int) * inSize);
        int *expected = malloc(sizeof(int) * inSize);
        int *out = malloc(sizeof(int) * inSize);
        memset(out, 0, sizeof(int) * inSize);
        for(int i = 0; i < inSize; ++i) {
                a[i] = i + 1;
                b[i] = i - 1;
                expected[i] = 2 * i;
        }

        printf("Running on DFE.\n");
        TRPO(inSize, a, b, out);


  /***
      Note that you should always test the output of your DFE
      design against a CPU version to ensure correctness.
  */

        for (int i = 0; i < inSize; i++)
                if (out[i] != expected[i]) {
                        printf("Output from DFE did not match CPU: %d : %d != %d\n", i, out[i], expected[i]);
                        return 1;
                }

        printf("Test passed!\n");
        return 0;
}
