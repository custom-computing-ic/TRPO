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
    // Activation Function of each layer: 't' = tanh, 'l' = linear (y=x), 's' = sigmoid
    // Activation Function used in the Final Layer in modular_rl: 'o' y = 0.1x
    char * AcFunc;
    // Number of nodes in each layer: from [Input] to [Output]
    // InvertedPendulum-v1: 4, 64, 64, 1
    // Humanoid-v1: 376, 64, 64, 17    
    size_t * LayerSize;
    // Number of Samples
    size_t NumSamples;
    // Conjugate Gradient Damping Factor 
    double CG_Damping;
    
} TRPOparam;


size_t NumParamsCalc (size_t * LayerSize, size_t NumLayers) {

    // Utility function calculating the number of trainable parameters
    size_t NumParams = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        // Weight and Bias
        NumParams += LayerSize[i] * LayerSize[i+1] + LayerSize[i+1];
    }
    // Std
    NumParams += LayerSize[NumLayers-1];
    return NumParams;
}


int FVP (TRPOparam param, double *Result, double *Input) 
{

    //////////////////// Remarks ////////////////////
    
    // This function computes the Fisher-Vector Product using Pearlmutter Algorithm
    // Input: the vector to be multiplied with the Fisher Information Matrix     
    // Result: the Fisher-Vector Product
    // Remarks: The length of Input and Result must be the number of all trainable parameters in the network

    // Step1: ordinary forward propagation
    // Step2: ordinary backward propagation
    // Step3: Pearlmutter forward propagation
    // Step4: Pearlmutter backward propagation


    //////////////////// Read Parameters ////////////////////

    // Assign Parameters
    const size_t NumLayers  = param.NumLayers;
    char * AcFunc           = param.AcFunc;
    size_t * LayerSize      = param.LayerSize;
    const size_t NumSamples = param.NumSamples;
    char * ModelFile        = param.ModelFile;
    char * DataFile         = param.DataFile;
    const double CG_Damping = param.CG_Damping;
    
    // Dimension of Observation Space
    const size_t ObservSpaceDim = LayerSize[0];
    // Dimension of Action Space
    const size_t ActionSpaceDim = LayerSize[NumLayers-1];
    
    // iterator when traversing through input vector and result vector
    size_t pos;


    //////////////////// Memory Allocation - Neural Network ////////////////////
    
    // W[i]: Weight Matrix from Layer[i] to Layer[i+1]
    // B[i]: Bias Vector from Layer[i] to Layer[i+1]
    // Item (j,k) in W[i] refers to the weight from Neuron #j in Layer[i] to Neuron #k in Layer[i+1]
    // Item B[k] is the bias of Neuron #k in Layer[i+1]
    double * W [NumLayers-1];
    double * B [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        W[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        B[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }

    // LogStd[i]: log standard deviation for action dimension #i in the Diagonal Gaussian Distribution
    double * LogStd = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Memory Allocation - Input Vector ////////////////////

    // The Input Vector is to be multiplied with the Hessian Matrix of KL to derive the Fisher Vector Product
    // There is one-to-one correspondence between the input vector and all trainable parameters in the neural network
    // As a result, the shape of the Input Vector is the same as that of the parameters in the model
    // The only difference is that the Input Vector is stored in a flattened manner
    // There is one-to-one correspondence between: VW[i] and W[i], VB[i] and B[i], VStd[i] and Std[i]
    double * VW [NumLayers-1];
    double * VB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        VW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        VB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }
    
    // Allocate Memory for Input Vector corresponding to LogStd
    double * VLogStd = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Memory Allocation - Simulation Data ////////////////////

    // Allocate Memory for Observation and Probability Mean
    // Observ: list of observations - corresponds to ob_no in modular_rl
    // Mean: list of probablity mean values - corresponds to the 'mean' part of prob_np in modular_rl
    // Remarks: due to the specific setting of the experienments in the TRPO paper, 
    //          Std is the same for all samples in each simulation iteration,
    //          so we just allocate Std memory space for one sample and use it for all samples.
    //          The general case should be another vector of Std with size NumSamples*ActionSpaceDim
    double * Observ = (double *) calloc(NumSamples*ObservSpaceDim, sizeof(double));
    double * Mean   = (double *) calloc(NumSamples*ActionSpaceDim, sizeof(double));
    double * Std    = (double *) calloc(ActionSpaceDim, sizeof(double));

    // Allocate Memory for Average Sample Mean and Average Sample Mean Square
    // Remarks: These values are statistics calculated from the samples, to be used in the algorithm
    double * AvgSampleMean   = (double *) calloc(ActionSpaceDim, sizeof(double));
    double * AvgSampleMeanSq = (double *) calloc(ActionSpaceDim, sizeof(double));
    
    
    //////////////////// Memory Allocation - Ordinary Forward and Backward Propagation ////////////////////

    // Layer[i] : Memory of each layer's outputs, i.e. y_i
    // GLayer[I]: Gradient of KL w.r.t. the pre-activation values in Layer[i], i.e. d(KL)/d(x_i)
    double * Layer  [NumLayers];
    double * GLayer [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        Layer[i]  = (double *) calloc(LayerSize[i], sizeof(double));
        GLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
    }

    // GW[i]: Gradient of KL w.r.t to Neural Network Weight W[i]
    // GB[i]: Gradient of KL w.r.t to Neural Network Bias B[i]
    // There is one-to-one correspondence between: GW[i] and W[i], GB[i] and B[i], GStd[i] and Std[i]
    double * GW [NumLayers-1];
    double * GB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        GW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        GB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }

    // GStd[i]: Gradient of KL w.r.t standard deviation Std[i]
    double * GStd = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Memory Allocation - Pearlmutter Forward and Backward Propagation ////////////////////

    // RyLayer[i]: R{} of each layer's outputs, i.e. R{y_i}
    // RxLayer[i]: R{} of each layer's pre-activated outputs, i.e. R{x_i}
    // RGLayer[I]: R{} Gradient of KL w.r.t. the pre-activation values in Layer[i], i.e. R{d(KL)/d(x_i)}
    double * RyLayer [NumLayers];
    double * RxLayer [NumLayers];
    double * RGLayer [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        RyLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
        RxLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
        RGLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
    }

    // RGW[i]: R{} Gradient of KL w.r.t. to Neural Network Weight W[i], i.e. R{d(KL)/d(W[i])}
    // RGB[i]: R{} Gradient of KL w.r.t. to Neural Network Bias B[i], i.e. R{d(KL)/d(B[i])}
    // There is one-to-one correspondence between: RGW[i] and W[i], RGB[i] and B[i]
    double * RGW [NumLayers-1];
    double * RGB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        RGW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        RGB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }

    //  RStd[i]: R{} of Std[i], i.e. R{Std[i]}
    // RGStd[i]: R{} Gradient of KL w.r.t. log standard deviation LogStd[i], i.e. R{d(KL)/d(LogStd[i])}
    double * RStd     = (double *) calloc(ActionSpaceDim, sizeof(double));
    double * RGLogStd = (double *) calloc(ActionSpaceDim, sizeof(double));
    
    
    //////////////////// Load Neural Network ////////////////////
    
    // Open Model File that contains Weights, Bias and std
	FILE *ModelFilePointer = fopen(ModelFile, "r");
	if (ModelFilePointer==NULL) {
		fprintf(stderr, "[ERROR] Cannot open Model File [%s]. \n", ModelFile);
  		return -1;
	}    
    
    // Read Weights and Bias from file
    for (size_t i=0; i<NumLayers-1; ++i) {
        // Reading Weights W[i]: from Layer[i] to Layer[i+1]
        size_t curLayerDim  = LayerSize[i];
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

    // Read LogStd from file
    // Remarks: actually this std will be overwritten by the std from the datafile
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        fscanf(ModelFilePointer, "%lf", &LogStd[k]);
    }

    // Close Model File
    fclose(ModelFilePointer);
    
    
    //////////////////// Load Input Vector and Init Result Vector ////////////////////    
    
    pos = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        size_t curLayerDim  = LayerSize[i];
        size_t nextLayerDim = LayerSize[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                VW[i][j*nextLayerDim+k] = Input[pos];
                Result[pos] = 0;
                pos++;
            }
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            VB[i][k] = Input[pos];
            Result[pos] = 0;
            pos++;
        }
    }
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        VLogStd[k] = Input[pos];
        Result[pos] = 0;
        pos++;
    }
    
    
    //////////////////// Load Simulation Data ////////////////////
    
    // Open Data File that contains Mean, std and Observation
	FILE *DataFilePointer = fopen(DataFile, "r");
	if (DataFilePointer==NULL) {
		fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", DataFile);
  		return -1;
	}  
    
    // Read Mean, Std and Observation
    // Remarks: Std is the same for all samples, and appears in every line in the data file
    //          so we are reading the same Std again and again to the same place.
    for (size_t i=0; i<NumSamples; ++i) {
        // Read Mean
        for (size_t j=0; j<ActionSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Mean[i*ActionSpaceDim+j]);
        }
        // Read Std
        for (size_t j=0; j<ActionSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Std[j]);
        }
        // Read Observation
        for (size_t j=0; j<ObservSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Observ[i*ObservSpaceDim+j]);
        }
    }
    
    // Close Data File
    fclose(DataFilePointer);

    // Compute Average Sample Mean and Average Sample Mean Square
    for (size_t i=0; i<NumSamples; ++i) {
        for (size_t j=0; j<ActionSpaceDim; ++j) {
            AvgSampleMean[j]   += Mean[i*ActionSpaceDim+j];
            AvgSampleMeanSq[j] += Mean[i*ActionSpaceDim+j] * Mean[i*ActionSpaceDim+j];
        }
    }
    for (size_t j=0; j<ActionSpaceDim; ++j) {
        AvgSampleMean[j]   = AvgSampleMean[j]   / (double)NumSamples;
        AvgSampleMeanSq[j] = AvgSampleMeanSq[j] / (double)NumSamples;
    }
    
    
    //////////////////// Main Loop Over All Samples ////////////////////
    
    for (size_t iter=0; iter<NumSamples; iter++) {
    
        //////////////////// Ordinary Forward Propagation ////////////////////
    
        // Assign Input Values
        for (size_t i=0; i<ObservSpaceDim; ++i) Layer[0][i] = Observ[iter*ObservSpaceDim+i];
    
        // Forward Propagation
        for (size_t i=0; i<NumLayers-1; ++i) {
            
            // Propagate from Layer[i] to Layer[i+1]
            for (size_t j=0; j<LayerSize[i+1]; ++j) {
                
                // Calculating pre-activated value for item[j] in next layer
                Layer[i+1][j] = B[i][j];
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                    Layer[i+1][j] += Layer[i][k] * W[i][k*LayerSize[i+1]+j];
                }
            
                // Apply Activation Function
                switch (AcFunc[i+1]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {break;}
                    // tanh() Activation Function
                    case 't': {Layer[i+1][j] = tanh(Layer[i+1][j]); break;}
                    // 0.1x Activation Function
                    case 'o': {Layer[i+1][j] = 0.1*Layer[i+1][j]; break;}
                    // sigmoid Activation Function
                    case 's': {Layer[i+1][j] = 1.0/(1+exp(-Layer[i+1][j])); break;}
                    // Default: Activation Function not supported
                    default: {
                        printf("[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                        return -1;
                    }
                }
            }
        }

        // Check whether the forward propagation output is correct
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            double output   = Layer[NumLayers-1][i];
            double expected = Mean[iter*ActionSpaceDim+i];
            double err      = abs( (output - expected) / expected ) * 100;
            if (err>0.1) printf("out[%zu] = %e, mean = %e => %.4f%% Difference\n", i, output, expected, err);
        }

        
        //////////////////// Ordinary Backward Propagation ////////////////////         

        // Gradient Initialisation
        // Assign the derivative of KL w.r.t. Mean (output values from the final layer) and Std
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            GLayer[NumLayers-1][i] = 0;
            GStd[i] = 0;
        }
        
//        The two equation below is for the SimpleModel2-2-2
//        GLayer[NumLayers-1][0] = Layer[NumLayers-1][0] - 0.01;
//        GLayer[NumLayers-1][1] = Layer[NumLayers-1][1] - 0.99;

        // Backward Propagation
        for (size_t i=NumLayers-1; i>0; --i) {
       
            // Propagate from Layer[i] to Layer[i-1]
            for (size_t j=0; j<LayerSize[i]; ++j) {

                // Differentiate the activation function
                switch (AcFunc[i]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {break;}
                    // tanh() Activation Function: tanh' = 1 - tanh^2
                    case 't': {GLayer[i][j] = GLayer[i][j] * (1- Layer[i][j] * Layer[i][j]); break;}
                    // 0.1x Activation Function
                    case 'o': {GLayer[i][j] = 0.1 * GLayer[i][j]; break;}
                    // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                    case 's': {GLayer[i][j] = GLayer[i][j] * Layer[i][j] * (1- Layer[i][j]); break;}
                    // Default: Activation Function not supported
                    default: {
                        fprintf(stderr, "[ERROR] Activation Function for Layer[%zu] is %c. Unsupported.\n", i, AcFunc[i]);
                        return -1;
                    }
                }
                
                // The derivative w.r.t to Bias is the same as that w.r.t. the pre-activated value
                GB[i-1][j] = GLayer[i][j];
            }
        
            // Calculate the derivative w.r.t. to Weight
            for (size_t j=0; j<LayerSize[i-1]; ++j) {
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // The Derivative w.r.t. to the weight from Neuron #j in Layer[i-1] to Neuron #k in Layer[i]
                    GW[i-1][j*LayerSize[i]+k] = GLayer[i][k] * Layer[i-1][j];
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

        
        //////////////////// Pearlmutter Forward Propagation ////////////////////           

        
        // Input is constant, so the R{} derivative is 0 
        for (size_t i=0; i<ObservSpaceDim; ++i) {
            RyLayer[0][i] = 0;
            RxLayer[0][i] = 0;
        }
    
        // Forward Propagation
        for (size_t i=0; i<NumLayers-1; ++i) {            
            
            // Propagate from Layer[i] to Layer[i+1]
            for (size_t j=0; j<LayerSize[i+1]; ++j) {
                
                // Calculate R{x_j} in next layer
                RxLayer[i+1][j] = VB[i][j];
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                    RxLayer[i+1][j] += RyLayer[i][k] * W[i][k*LayerSize[i+1]+j];
                    RxLayer[i+1][j] += Layer[i][k] * VW[i][k*LayerSize[i+1]+j];
                }

                // Calculate R{y_j} in next layer, need to differentiate Activation Function
                switch (AcFunc[i+1]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {RyLayer[i+1][j] = RxLayer[i+1][j]; break;}
                    // tanh() Activation Function: tanh' = 1 - tanh^2
                    case 't': {RyLayer[i+1][j] = RxLayer[i+1][j] * (1- Layer[i+1][j] * Layer[i+1][j]); break;}
                    // 0.1x Activation Function
                    case 'o': {RyLayer[i+1][j] = 0.1 * RxLayer[i+1][j]; break;}
                    // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                    case 's': {RyLayer[i+1][j] = RxLayer[i+1][j] * Layer[i+1][j] * (1- Layer[i+1][j]); break;}
                    // Default: Activation Function not supported
                    default: {
                        fprintf(stderr, "[ERROR] Activation Function for Layer[%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                        return -1;
                    }
                }
            }
        }
        
        // Calculating R{Std}
        // Remarks: R{Std} is w.r.t. to Std. 
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            RStd[i] = Std[i] * VLogStd[i];
        }

        //////////////////// Pearlmutter Backward Propagation ////////////////////


        // Gradient Initialisation
        // Calculating R{} Gradient of KL w.r.t. output values from the final layer, i.e. R{d(KL)/d(mean_i)}
        // Calculating R{} Gradient of KL w.r.t. LogStd, i.e. R{d(KL)/d(LogStd[i])}
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            double StdSq = Std[i] * Std[i];
            RGLayer[NumLayers-1][i] = RyLayer[NumLayers-1][i]/StdSq - 2*GLayer[NumLayers-1][i]/Std[i]*RStd[i];
            RGLogStd[i] = 2*RStd[i]/Std[i];
        }


        // Backward Propagation
        for (size_t i=NumLayers-1; i>0; --i) {
       
            // Propagate from Layer[i] to Layer[i-1]
            for (size_t j=0; j<LayerSize[i]; ++j) {

                // Calculating R{} Gradient of KL w.r.t. pre-activated values in Layer[i], i.e. R{d(KL)/d(x_i)}
                // Differentiate the activation function
                switch (AcFunc[i]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {break;}
                    // tanh() Activation Function: tanh' = 1 - tanh^2
                    case 't': {
                        RGLayer[i][j] = (1-Layer[i][j]*Layer[i][j])*RGLayer[i][j] - 2*Layer[i][j]*GLayer[i][j]*RxLayer[i][j];
                        break;
                    }
                    // 0.1x Activation Function
                    case 'o': {RGLayer[i][j] = 0.1 * RGLayer[i][j]; break;}
                    // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                    case 's': {
                        RGLayer[i][j] = RGLayer[i][j]*Layer[i][j]*(1-Layer[i][j]) + GLayer[i][j]*(1-2*Layer[i][j])*RxLayer[i][j]; 
                        break;
                    }
                    // Default: Activation Function not supported
                    default: {
                        fprintf(stderr, "[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i, AcFunc[i]);
                        return -1;
                    }
                }
                
                // The R{} derivative w.r.t to Bias is the same as that w.r.t. the pre-activated value
                RGB[i-1][j] = RGLayer[i][j];
            }

            // Calculate the R{} derivative w.r.t. to Weight
            for (size_t j=0; j<LayerSize[i-1]; ++j) {
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // The R{} Derivative w.r.t. to the weight from Neuron #j in Layer[i-1] to Neuron #k in Layer[i]
                    RGW[i-1][j*LayerSize[i]+k] = Layer[i-1][j] * RGLayer[i][k] + RyLayer[i-1][j] * GLayer[i][k];
                }
            }

            // Calculate the R{} derivative w.r.t. the output values from Layer[i]
            for (size_t j=0; j<LayerSize[i-1]; ++j) {
                RGLayer[i-1][j] = 0;
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // Accumulate the Gradient from Neuron #k in Layer[i] to Neuron #j in Layer[i-1]
                    RGLayer[i-1][j] += VW[i-1][j*LayerSize[i]+k] * GLayer[i][k];
                    RGLayer[i-1][j] += W[i-1][j*LayerSize[i]+k] * RGLayer[i][k];
                }
            }
        }
        
        // Accumulate the Fisher-Vector Product to result
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim = LayerSize[i];
            size_t nextLayerDim = LayerSize[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    Result[pos] += RGW[i][j*nextLayerDim+k];
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                Result[pos] += RGB[i][k];
                pos++;
            }
        }
        for (size_t k=0; k<ActionSpaceDim; ++k) {
            Result[pos] += RGLogStd[k];
            pos++;
        }

    
    } // End of iteration over current sample


    // Averaging Fisher Vector Product over the samples and apply CG Damping
    for (size_t i=0; i<pos; ++i) {
        Result[i] = Result[i] / (double)NumSamples;
        Result[i] += CG_Damping * Input[i];
    }


    //////////////////// Clean Up ////////////////////  

    // clean up
    for (size_t i=0; i<NumLayers; ++i) {
        free(Layer[i]); free(GLayer[i]);
        free(RyLayer[i]); free(RxLayer[i]); free(RGLayer[i]);
    }
    for (size_t i=0; i<NumLayers-1; ++i) {
        free(W[i]); free(VW[i]); free(GW[i]); free(RGW[i]);
        free(B[i]); free(VB[i]); free(GB[i]); free(RGB[i]);
    }
    free(LogStd); free(VLogStd); free(RGLogStd);
    free(GStd); free(RStd);
    free(Observ); free(Mean); free(Std);
    free(AvgSampleMean); free(AvgSampleMeanSq);

    return 0;
}


int FVPFast (TRPOparam param, double *Result, double *Input)
{

    //////////////////// Remarks ////////////////////

    // This function computes the Fisher-Vector Product using Pearlmutter Algorithm
    // This version is customised to the case that KL is used as loss function
    // Input: the vector to be multiplied with the Fisher Information Matrix
    // Result: the Fisher-Vector Product
    // Remarks: The length of Input and Result must be the number of all trainable parameters in the network

    // Step1: ordinary forward propagation
    // Step2: Pearlmutter forward propagation
    // Step3: Pearlmutter backward propagation


    //////////////////// Read Parameters ////////////////////

    // Assign Parameters
    const size_t NumLayers  = param.NumLayers;
    char * AcFunc           = param.AcFunc;
    size_t * LayerSize      = param.LayerSize;
    const size_t NumSamples = param.NumSamples;
    char * ModelFile        = param.ModelFile;
    char * DataFile         = param.DataFile;
    const double CG_Damping = param.CG_Damping;

    // Dimension of Observation Space
    const size_t ObservSpaceDim = LayerSize[0];
    // Dimension of Action Space
    const size_t ActionSpaceDim = LayerSize[NumLayers-1];

    // iterator when traversing through input vector and result vector
    size_t pos;


    //////////////////// Memory Allocation - Neural Network ////////////////////
    
    // W[i]: Weight Matrix from Layer[i] to Layer[i+1]
    // B[i]: Bias Vector from Layer[i] to Layer[i+1]
    // Item (j,k) in W[i] refers to the weight from Neuron #j in Layer[i] to Neuron #k in Layer[i+1]
    // Item B[k] is the bias of Neuron #k in Layer[i+1]
    double * W [NumLayers-1];
    double * B [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        W[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        B[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }


    //////////////////// Memory Allocation - Input Vector ////////////////////

    // The Input Vector is to be multiplied with the Hessian Matrix of KL to derive the Fisher Vector Product
    // There is one-to-one correspondence between the input vector and all trainable parameters in the neural network
    // As a result, the shape of the Input Vector is the same as that of the parameters in the model
    // The only difference is that the Input Vector is stored in a flattened manner
    // There is one-to-one correspondence between: VW[i] and W[i], VB[i] and B[i], VStd[i] and Std[i]
    double * VW [NumLayers-1];
    double * VB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        VW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        VB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }
    
    // Allocate Memory for Input Vector corresponding to LogStd
    double * VLogStd = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Memory Allocation - Simulation Data ////////////////////

    // Allocate Memory for Observation and Probability Mean
    // Observ: list of observations - corresponds to ob_no in modular_rl
    // Mean: list of probablity mean values - corresponds to the 'mean' part of prob_np in modular_rl
    // Remarks: due to the specific setting of the experienments in the TRPO paper,
    //          Std is the same for all samples in each simulation iteration,
    //          so we just allocate Std memory space for one sample and use it for all samples.
    //          The general case should be another vector of Std with size NumSamples*ActionSpaceDim
    double * Observ = (double *) calloc(NumSamples*ObservSpaceDim, sizeof(double));
    double * Mean   = (double *) calloc(NumSamples*ActionSpaceDim, sizeof(double));
    double * Std    = (double *) calloc(ActionSpaceDim, sizeof(double));
    
    
    //////////////////// Memory Allocation - Ordinary Forward Propagation ////////////////////

    // Layer[i] : Memory of each layer's outputs, i.e. y_i
    double * Layer  [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        Layer[i]  = (double *) calloc(LayerSize[i], sizeof(double));
    }


    //////////////////// Memory Allocation - Pearlmutter Forward and Backward Propagation ////////////////////

    // RyLayer[i]: R{} of each layer's outputs, i.e. R{y_i}
    // RxLayer[i]: R{} of each layer's pre-activated outputs, i.e. R{x_i}
    // RGLayer[I]: R{} Gradient of KL w.r.t. the pre-activation values in Layer[i], i.e. R{d(KL)/d(x_i)}
    double * RyLayer [NumLayers];
    double * RxLayer [NumLayers];
    double * RGLayer [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        RyLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
        RxLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
        RGLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
    }

    // RGW[i]: R{} Gradient of KL w.r.t. to Neural Network Weight W[i], i.e. R{d(KL)/d(W[i])}
    // RGB[i]: R{} Gradient of KL w.r.t. to Neural Network Bias B[i], i.e. R{d(KL)/d(B[i])}
    // There is one-to-one correspondence between: RGW[i] and W[i], RGB[i] and B[i]
    double * RGW [NumLayers-1];
    double * RGB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        RGW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        RGB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }
    
    
    //////////////////// Load Neural Network ////////////////////
    
    // Open Model File that contains Weights, Bias and std
    FILE *ModelFilePointer = fopen(ModelFile, "r");
    if (ModelFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Model File [%s]. \n", ModelFile);
        return -1;
    }
    
    // Read Weights and Bias from file
    for (size_t i=0; i<NumLayers-1; ++i) {
        // Reading Weights W[i]: from Layer[i] to Layer[i+1]
        size_t curLayerDim  = LayerSize[i];
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

    // Read LogStd from file
    // Remarks: actually this LogStd will be overwritten by the Std from the datafile
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        fscanf(ModelFilePointer, "%lf", &Std[k]);
    }

    // Close Model File
    fclose(ModelFilePointer);
    
    
    //////////////////// Load Input Vector and Init Result Vector ////////////////////
    
    pos = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        size_t curLayerDim  = LayerSize[i];
        size_t nextLayerDim = LayerSize[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                VW[i][j*nextLayerDim+k] = Input[pos];
                Result[pos] = 0;
                pos++;
            }
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            VB[i][k] = Input[pos];
            Result[pos] = 0;
            pos++;
        }
    }
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        VLogStd[k] = Input[pos];
        Result[pos] = 0;
        pos++;
    }
    
    
    //////////////////// Load Simulation Data ////////////////////
    
    // Open Data File that contains Mean, std and Observation
    FILE *DataFilePointer = fopen(DataFile, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", DataFile);
        return -1;
    }
    
    // Read Mean, Std and Observation
    // Remarks: Std is the same for all samples, and appears in every line in the data file
    //          so we are writing the same Std again and again to the same place.
    for (size_t i=0; i<NumSamples; ++i) {
        // Read Mean
        for (size_t j=0; j<ActionSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Mean[i*ActionSpaceDim+j]);
        }
        // Read Std
        for (size_t j=0; j<ActionSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Std[j]);
        }
        // Read Observation
        for (size_t j=0; j<ObservSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Observ[i*ObservSpaceDim+j]);
        }
    }
    
    // Close Data File
    fclose(DataFilePointer);
    
    
    //////////////////// Main Loop Over All Samples ////////////////////
    
    for (size_t iter=0; iter<NumSamples; iter++) {
    
        //////////////////// Ordinary Forward Propagation ////////////////////
    
        // Assign Input Values
        for (size_t i=0; i<ObservSpaceDim; ++i) Layer[0][i] = Observ[iter*ObservSpaceDim+i];
    
        // Forward Propagation
        for (size_t i=0; i<NumLayers-1; ++i) {
            
            // Propagate from Layer[i] to Layer[i+1]
            for (size_t j=0; j<LayerSize[i+1]; ++j) {
                
                // Calculating pre-activated value for item[j] in next layer
                Layer[i+1][j] = B[i][j];
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                    Layer[i+1][j] += Layer[i][k] * W[i][k*LayerSize[i+1]+j];
                }
            
                // Apply Activation Function
                switch (AcFunc[i+1]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {break;}
                    // tanh() Activation Function
                    case 't': {Layer[i+1][j] = tanh(Layer[i+1][j]); break;}
                    // 0.1x Activation Function
                    case 'o': {Layer[i+1][j] = 0.1*Layer[i+1][j]; break;}
                    // sigmoid Activation Function
                    case 's': {Layer[i+1][j] = 1.0/(1+exp(-Layer[i+1][j])); break;}
                    // Default: Activation Function not supported
                    default: {
                        printf("[ERROR] AC Function for Layer[%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                        return -1;
                    }
                }
            }
        }

        // Check whether the forward propagation output is correct
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            double output   = Layer[NumLayers-1][i];
            double expected = Mean[iter*ActionSpaceDim+i];
            double err      = abs( (output - expected) / expected ) * 100;
            if (err>0.1) printf("out[%zu] = %e, mean = %e => %.4f%% Difference\n", i, output, expected, err);
        }

        
        //////////////////// Pearlmutter Forward Propagation ////////////////////

        
        // Input is constant, so the R{} derivative is 0
        for (size_t i=0; i<ObservSpaceDim; ++i) {
            RyLayer[0][i] = 0;
            RxLayer[0][i] = 0;
        }
    
        // Forward Propagation
        for (size_t i=0; i<NumLayers-1; ++i) {
            
            // Propagate from Layer[i] to Layer[i+1]
            for (size_t j=0; j<LayerSize[i+1]; ++j) {
                
                // Calculate R{x_j} in next layer
                RxLayer[i+1][j] = VB[i][j];
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                    RxLayer[i+1][j] += RyLayer[i][k] * W[i][k*LayerSize[i+1]+j];
                    RxLayer[i+1][j] += Layer[i][k] * VW[i][k*LayerSize[i+1]+j];
                }

                // Calculate R{y_j} in next layer, need to differentiate Activation Function
                switch (AcFunc[i+1]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {RyLayer[i+1][j] = RxLayer[i+1][j]; break;}
                    // tanh() Activation Function: tanh' = 1 - tanh^2
                    case 't': {RyLayer[i+1][j] = RxLayer[i+1][j] * (1- Layer[i+1][j] * Layer[i+1][j]); break;}
                    // 0.1x Activation Function
                    case 'o': {RyLayer[i+1][j] = 0.1 * RxLayer[i+1][j]; break;}
                    // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                    case 's': {RyLayer[i+1][j] = RxLayer[i+1][j] * Layer[i+1][j] * (1- Layer[i+1][j]); break;}
                    // Default: Activation Function not supported
                    default: {
                        fprintf(stderr, "[ERROR] AC Function for Layer[%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                        return -1;
                    }
                }
            }
        }


        //////////////////// Pearlmutter Backward Propagation ////////////////////


        // Gradient Initialisation
        // Calculating R{} Gradient of KL w.r.t. output values from the final layer, i.e. R{d(KL)/d(mean_i)}
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            RGLayer[NumLayers-1][i] = RyLayer[NumLayers-1][i] / Std[i] / Std[i];
        }

        // Backward Propagation
        for (size_t i=NumLayers-1; i>0; --i) {
       
            // Propagate from Layer[i] to Layer[i-1]
            for (size_t j=0; j<LayerSize[i]; ++j) {

                // Calculating R{} Gradient of KL w.r.t. pre-activated values in Layer[i], i.e. R{d(KL)/d(x_i)}
                // Differentiate the activation function
                switch (AcFunc[i]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {break;}
                    // tanh() Activation Function: tanh' = 1 - tanh^2
                    case 't': {RGLayer[i][j] = (1-Layer[i][j]*Layer[i][j])*RGLayer[i][j]; break;}
                    // 0.1x Activation Function
                    case 'o': {RGLayer[i][j] = 0.1 * RGLayer[i][j]; break;}
                    // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                    case 's': {RGLayer[i][j] = RGLayer[i][j]*Layer[i][j]*(1-Layer[i][j]); break;}
                    // Default: Activation Function not supported
                    default: {
                        fprintf(stderr, "[ERROR] AC Function for Layer [%zu] is %c. Unsupported.\n", i, AcFunc[i]);
                        return -1;
                    }
                }
                
                // The R{} derivative w.r.t to Bias is the same as that w.r.t. the pre-activated value
                RGB[i-1][j] = RGLayer[i][j];
            }

            // Calculate the R{} derivative w.r.t. to Weight
            for (size_t j=0; j<LayerSize[i-1]; ++j) {
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // The R{} Derivative w.r.t. to the weight from Neuron #j in Layer[i-1] to Neuron #k in Layer[i]
                    RGW[i-1][j*LayerSize[i]+k] = Layer[i-1][j] * RGLayer[i][k];
                }
            }

            // Calculate the R{} derivative w.r.t. the output values from Layer[i]
            for (size_t j=0; j<LayerSize[i-1]; ++j) {
                RGLayer[i-1][j] = 0;
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // Accumulate the Gradient from Neuron #k in Layer[i] to Neuron #j in Layer[i-1]
                    RGLayer[i-1][j] += W[i-1][j*LayerSize[i]+k] * RGLayer[i][k];
                }
            }
        }
        
        // Accumulate the Fisher-Vector Product to result
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim = LayerSize[i];
            size_t nextLayerDim = LayerSize[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    Result[pos] += RGW[i][j*nextLayerDim+k];
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                Result[pos] += RGB[i][k];
                pos++;
            }
        }
        for (size_t k=0; k<ActionSpaceDim; ++k) {
            Result[pos] += 2 * VLogStd[k];
            pos++;
        }

    
    } // End of iteration over current sample


    // Averaging Fisher Vector Product over the samples and apply CG Damping
    for (size_t i=0; i<pos; ++i) {
        Result[i] = Result[i] / (double)NumSamples;
        Result[i] += CG_Damping * Input[i];
    }


    //////////////////// Clean Up ////////////////////

    // clean up
    for (size_t i=0; i<NumLayers; ++i) {
        free(Layer[i]); free(RxLayer[i]); free(RyLayer[i]); free(RGLayer[i]);
    }
    for (size_t i=0; i<NumLayers-1; ++i) {
        free(W[i]); free(VW[i]); free(RGW[i]);
        free(B[i]); free(VB[i]); free(RGB[i]);
    }
    free(Observ); free(Mean); free(Std); free(VLogStd);

    return 0;
}


int CG(TRPOparam param, double *Result, double *b, size_t MaxIter, double ResidualTh)
{

    //////////////////// Conjugate Gradient ////////////////////

    // This function implements Conjugate Gradient algorithm to solve linear equation Ax=b
    //     Result: The Conjugate Gradient Result, i.e. solution x to Ax=b
    //          b: Vector b in the equation Ax=b
    //    MaxIter: Maximum Iterations of Conjugate Gradient (in modular_rl is 10)
    // ResidualTh: Threshold of Residual (in modular_rl is 1e-10)

    // Memory Allocation
    size_t NumParams = NumParamsCalc(param.LayerSize, param.NumLayers);
    double * p = (double *) calloc(NumParams, sizeof(double));
    double * r = (double *) calloc(NumParams, sizeof(double));
    double * x = (double *) calloc(NumParams, sizeof(double));
    double * z = (double *) calloc(NumParams, sizeof(double));

    // Initialisation
    double rdotr = 0;
    for (size_t i=0; i<NumParams; ++i) {
        p[i] = b[i];
        r[i] = b[i];
        rdotr += r[i] * r[i];
    }
    
    // Iterative Solver
    for (size_t iter=0; iter<=MaxIter; ++iter) {

        // Calculate Frobenius Norm of x
        double FrobNorm = 0;
        for (size_t i=0; i<NumParams; ++i) FrobNorm += x[i] * x[i];
        FrobNorm = sqrt(FrobNorm);
        printf("CG Iter[%zu] Residual Norm=%e, Soln Norm=%e\n", iter, rdotr, FrobNorm);
        
        // Check Termination Condition
        if (rdotr<ResidualTh || iter==MaxIter) {
            for (size_t i=0; i<NumParams; ++i) Result[i] = x[i];
            break;
        }
        
        // Calculate z = FIM*p
        int FVPStatus = FVPFast(param, z, p);
        if (FVPStatus!=0) {
            fprintf(stderr, "[ERROR] Fisher Vector Product Calculation Failed.\n");
            free(p); free(r); free(x); free(z);
            return -1;
        }
        
        // Update x and r
        double pdotz = 0;
        for (size_t i=0; i<NumParams; ++i) pdotz += p[i] * z[i];
        double v = rdotr / pdotz;
        for (size_t i=0; i<NumParams; ++i) {
            x[i] += v * p[i];
            r[i] -= v * z[i];
        }
        
        // Update p
        double newrdotr = 0;
        for (size_t i=0; i<NumParams; ++i) newrdotr += r[i] * r[i];
        double mu = newrdotr / rdotr;
        for (size_t i=0; i<NumParams; ++i) p[i] = r[i] + mu * p[i];
        
        // Update rdotr
        rdotr = newrdotr;
    }
    
    // Clean Up
    free(p); free(r); free(x); free(z);
    
    return 0;
}


void SimpleTest() {

    // Simple Data 2-2-2
    char AcFunc [] = {'l', 's', 's'};
    size_t LayerSize [] = {2, 2, 2};

    TRPOparam Param;
    Param.ModelFile  = "SimpleTestModel.txt";
    Param.DataFile   = "SimpleTestData.txt";
    Param.NumLayers  = 3;
    Param.AcFunc     = AcFunc;
    Param.LayerSize  = LayerSize;
    Param.NumSamples = 1;
    Param.CG_Damping = 0.1;

    size_t NumParams = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double * input   = (double *) calloc(NumParams, sizeof(double));
    double * result  = (double *) calloc(NumParams, sizeof(double));

    int FVPStatus = FVP(Param, result, input);
    if (FVPStatus!=0) fprintf(stderr, "[ERROR] Fisher Vector Product Calculation Failed.\n");
    
    free(input); free(result);

}


void PendulumTest()
{
	
    // Simplified version of Pendulum-v0 with hidden_size set to 4
    char AcFunc [] = {'l', 't', 'l'};
    size_t LayerSize [] = {3, 4, 1};

    char * ModelFileName = "PendulumTestModel.txt";
    char * DataFileName  = "PendulumTestData.txt";
    char * FVPFileName   = "PendulumTestFVP.txt";

    TRPOparam Param;
    Param.ModelFile  = ModelFileName;
    Param.DataFile   = DataFileName;
    Param.NumLayers  = 3;
    Param.AcFunc     = AcFunc;
    Param.LayerSize  = LayerSize;
    Param.NumSamples = 200;
    Param.CG_Damping = 0.1;

    // Open Simulation Data File that contains test data
    FILE *DataFilePointer = fopen(FVPFileName, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", FVPFileName);
        return;
    }

    // Memory Allocation
    size_t NumParams = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double * input   = (double *) calloc(NumParams, sizeof(double));
    double * result  = (double *) calloc(NumParams, sizeof(double));
    double * expect  = (double *) calloc(NumParams, sizeof(double)); 
    
    // Read Input and Expect
    for (size_t i=0; i<NumParams; ++i) {
         fscanf(DataFilePointer, "%lf %lf", &input[i], &expect[i]);
    }
    fclose(DataFilePointer);

    int FVPStatus = FVPFast(Param, result, input);
    if (FVPStatus!=0) fprintf(stderr, "[ERROR] Fisher Vector Product Calculation Failed.\n");
    
    // Check Result
    printf("\n--------------------------- Pendulum Test ---------------------------");
    size_t pos = 0;
    for (size_t i=0; i<Param.NumLayers-1; ++i) {
        size_t curLayerDim = LayerSize[i];
        size_t nextLayerDim = LayerSize[i+1];
        printf("\n");
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                printf("FVP W[%zu][%zu][%zu] expected=%f, result=%f\n", i, j, k, expect[pos], result[pos]);
                pos++;
            }
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            printf("FVP Bias[%zu][%zu] expected=%f, result=%f\n", i, k, expect[pos], result[pos]);
            pos++;
        }
    }
    for (size_t k=0; k<LayerSize[Param.NumLayers-1]; ++k) {
        printf("\nFVP LogStd[%zu] expected=%f, result=%f\n", k, expect[pos], result[pos]);
        pos++;
    }
    
    double percentage_err = 0;
    for (size_t i=0; i<NumParams; ++i) {
        double cur_percentage_err = abs((result[i]-expect[i])/expect[i]);
    	if (expect[i] != 0) percentage_err += cur_percentage_err;
    }
    percentage_err = percentage_err * 100.0 / (double)NumParams;
    printf("\n[INFO] Fisher Vector Product Average Percentage Error = %.4f%%\n", percentage_err);
    printf("---------------------------------------------------------------------\n\n");

    // Clean Up    
    free(input); free(result); free(expect);
    
    return;
}

void SwimmerTest()
{
	
    // Swimmer-v1
    char AcFunc [] = {'l', 't', 't', 'l'};
    size_t LayerSize [] = {8, 64, 64, 2};

    char * ModelFileName = "SwimmerTestModel.txt";
    char * DataFileName  = "SwimmerTestData.txt";
    char * FVPFileName   = "SwimmerTestFVP.txt";

    TRPOparam Param;
    Param.ModelFile  = ModelFileName;
    Param.DataFile   = DataFileName;
    Param.NumLayers  = 4;
    Param.AcFunc     = AcFunc;
    Param.LayerSize  = LayerSize;
    Param.NumSamples = 26000;
    Param.CG_Damping = 0.1;

    // Open Simulation Data File that contains test data
    FILE *DataFilePointer = fopen(FVPFileName, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", FVPFileName);
        return;
    }

    // Memory Allocation
    size_t NumParams = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double * input   = (double *) calloc(NumParams, sizeof(double));
    double * result  = (double *) calloc(NumParams, sizeof(double));
    double * expect  = (double *) calloc(NumParams, sizeof(double)); 
    
    // Read Input and Expect
    for (size_t i=0; i<NumParams; ++i) {
         fscanf(DataFilePointer, "%lf %lf", &input[i], &expect[i]);
    }
    fclose(DataFilePointer);

    int FVPStatus = FVPFast(Param, result, input);
    if (FVPStatus!=0) fprintf(stderr, "[ERROR] Fisher Vector Product Calculation Failed.\n");
    
    // Check Result
    double percentage_err = 0;
    for (size_t i=0; i<NumParams; ++i) {        
        double cur_err = abs( (result[i]-expect[i])/expect[i] ) * 100;
    	if (expect[i] != 0) percentage_err += cur_err;
    	if (cur_err>0.1) printf("FVP[%zu]=%e, Expect=%e. %.4f%% Difference\n", i, result[i], expect[i], cur_err);
    }
    percentage_err = percentage_err / (double)NumParams;
    printf("--------------------------- Swimmer Test ----------------------------\n");
    printf("[INFO] Fisher Vector Product Average Percentage Error = %.4f%%\n", percentage_err);
    printf("---------------------------------------------------------------------\n\n");

    // Clean Up    
    free(input); free(result); free(expect);
    
    return;
}


void SwimmerCGTest()
{
	
    // Swimmer-v1
    char AcFunc [] = {'l', 't', 't', 'l'};
    size_t LayerSize [] = {8, 64, 64, 2};

    char * ModelFileName = "SwimmerTestModel.txt";
    char * DataFileName  = "SwimmerTestData.txt";
    char * CGFileName    = "SwimmerTestCG.txt";

    TRPOparam Param;
    Param.ModelFile  = ModelFileName;
    Param.DataFile   = DataFileName;
    Param.NumLayers  = 4;
    Param.AcFunc     = AcFunc;
    Param.LayerSize  = LayerSize;
    Param.NumSamples = 26000;
    Param.CG_Damping = 0.1;

    // Open Simulation Data File that contains test data
    FILE *DataFilePointer = fopen(CGFileName, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", CGFileName);
        return;
    }

    // Memory Allocation
    size_t NumParams = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double * input   = (double *) calloc(NumParams, sizeof(double));
    double * result  = (double *) calloc(NumParams, sizeof(double));
    double * expect  = (double *) calloc(NumParams, sizeof(double)); 
    
    // Read Input and Expect
    for (size_t i=0; i<NumParams; ++i) {
         fscanf(DataFilePointer, "%lf %lf", &input[i], &expect[i]);
    }
    fclose(DataFilePointer);
    
    printf("-------------------------- Swimmer CG Test ---------------------------\n");
    int CGStatus = CG(Param, result, input, 10, 1e-10);
    if (CGStatus!=0) fprintf(stderr, "[ERROR] Conjugate Gradient Calculation Failed.\n");
    
    // Check Result
    double percentage_err = 0;
    for (size_t i=0; i<NumParams; ++i) {        
        double cur_err = abs( (result[i]-expect[i])/expect[i] ) * 100;
    	if (expect[i] != 0) percentage_err += cur_err;
    	if (cur_err>0.1) printf("CG[%zu]=%e, Expect=%e. %.4f%% Difference\n", i, result[i], expect[i], cur_err);
    }
    percentage_err = percentage_err / (double)NumParams;
    printf("\n[INFO] Conjugate Gradient Average Percentage Error = %.4f%%\n", percentage_err);
    printf("---------------------------------------------------------------------\n\n");

    // Clean Up    
    free(input); free(result); free(expect);
    
    return;
}


int main()
{

    //////////////////// Fisher Vector Product Computation ////////////////////
    
    SimpleTest();
    PendulumTest();
    SwimmerTest();
    SwimmerCGTest();


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
