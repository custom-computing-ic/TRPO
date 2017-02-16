#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"

int main()
{

        //////////////////// C Version of Back Propagation ////////////////////
        
        //////////////////// Memory Allocation ////////////////////
        
        // Number of Layers in the network
        // e.g.: [Input] -----> [Hidden 1] -----> [Hidden 2] -----> [Output] is 4 layers.
        const size_t NumLayers = 3;
        // Activation Function of each layer: t = tanh, l = linear (activate(x)=x), s = sigmoid
        const char AcFunc [] = {'l', 's', 's'};
        // Number of Nodes: from [Input] to [Output]
        // InvertedPendulum-v1: 4, 64, 64, 1
        // Humanoid-v1: 367, 64, 64, 17
        const size_t LayerSize [] = {2, 2, 2};

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
        
        
        //////////////////// Initialisation ////////////////////
        
        // Assign Input Values to Layer[0]
//        for (size_t i=0; i<LayerSize[0]; ++i) Layer[0][i] = i;

        // ------------- Simple Test Case ------------- //
        // Simple Test Case: 2-2-2 Network with sigmoid
        
        // Assign Input Values
        Layer[0][0] = 0.05;
        Layer[0][1] = 0.10;
        
        // Assign Weights
        W[0][0*2+0] = 0.15;
        W[0][0*2+1] = 0.25;
        W[0][1*2+0] = 0.20;
        W[0][1*2+1] = 0.30;
        
        W[1][0*2+0] = 0.40;
        W[1][0*2+1] = 0.50;
        W[1][1*2+0] = 0.45;
        W[1][1*2+1] = 0.55;
        
        // Assign Bias
        B[0][0] = 0.35;
        B[0][1] = 0.35;
        B[1][0] = 0.60;
        B[1][1] = 0.60;
        
        // ------------- Simple Test Case ------------- //


        //////////////////// Forward Propagation ////////////////////
        
        for (size_t i=0; i<NumLayers-1; ++i) {
                // Propagate from Layer[i] to Layer[i+1]
                for (size_t j=0; j<LayerSize[i+1]; ++j) {
                        // Calculating item[j] in next layer
                        Layer[i+1][j] = B[i][j];
//                        printf("Net[%zu][%zu] init to bias %f \n", i+1, j, Layer[i+1][j]);
                        for (size_t k=0; k<LayerSize[i]; ++k) {
                                // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                                Layer[i+1][j] += Layer[i][k] * W[i][k*LayerSize[i+1]+j];
//                              printf("Net[%zu][%zu] += Input %f * Weight %f => %f \n", i+1, j, Layer[i][k], W[i][k*LayerSize[i+1]+j], Layer[i+1][j]);
                        }
//                        printf("Net[%zu][%zu] = %f => ", i+1, j, Layer[i+1][j]);
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
                                        exit(1);
                                }
                        }
//                        printf("Out[%zu][%zu] = %f \n", i+1, j, Layer[i+1][j]);
                }
        }
        
        // Output
        for (size_t i=0; i<LayerSize[NumLayers-1]; ++i) {
                printf("output[%zu] = %f \n", i, Layer[NumLayers-1][i]);
        }

        // TODO: *0.1 in the final layer for TRPO
        
        
        //////////////////// Backward Propagation ////////////////////        
        
        // TODO: Loss Function
        
        // Simple Test Case: 2-2-2 Network with sigmoid
        // Assign the derivative of Loss Func w.r.t. output layer
        //        for (size_t i=0; i<LayerSize[NumLayers-1]; ++i) GLayer[NumLayers-1][i] = i;
        
        // ------------- Simple Test Case ------------- //
        
        // Assign the derivative of Loss Func w.r.t. the output values from the final layer   
        GLayer[NumLayers-1][0] = Layer[NumLayers-1][0] - 0.01;
        GLayer[NumLayers-1][1] = Layer[NumLayers-1][1] - 0.99;
        
        for (size_t i=NumLayers-1; i>0; --i) {
                // Propagate from Layer[i] to Layer[i-1]
                
                // Calculate derivative of the activation function
                for (size_t j=0; j<LayerSize[i]; ++j) {
                        switch (AcFunc[i]) {
                                // Linear Activation Function: Ac(x) = (x)
                                case 'l': {break;}
                                // tanh() Activation Function: tanh' = 1 - tanh^2
                                case 't': {GLayer[i][j] = GLayer[i][j] * (1- Layer[i][j] * Layer[i][j]); break;}
                                // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                                case 's': {GLayer[i][j] = GLayer[i][j] * Layer[i][j] * (1- Layer[i][j]); break;}
                                // Default: Activation Function not supported
                                default: {
                                        printf("[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i+1, AcFunc[i]);
                                        exit(1);
                                }
                        }
                        // The derivative w.r.t to Bias is the same as that w.r.t. the pre-activated value
                        dB[i][j] = GLayer[i][j];
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
                
        }




        //////////////////// Clean Up ////////////////////  

        // clean up
        for (size_t i=0; i<NumLayers; ++i) {free(Layer[i]); free(GLayer[i]);}
        for (size_t i=0; i<NumLayers-1; ++i) {free(W[i]); free(dW[i]); free(B[i]); free(dB[i]);}



        // FPGA Stuff below

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
        Demo(inSize, a, b, out);


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
