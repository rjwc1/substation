#pragma once

#define sizeQ 3
#define sizeB 96
#define sizeS 128
#define sizeH 16
#define sizeP 64
#define sizeI 1024
#define sizeU 4096
#define sizeJ 128
#define sizeK 128

using Real = double;

#define ENCODER_SOFTMAX_DROPOUT_PROBABILITY 0.0

#define ENCODER_RESIDUAL1_DROPOUT_PROBABILITY 0.0

#define ENCODER_ACTIVATION_DROPOUT_PROBABILITY 0.0

#define ENCODER_RESIDUAL2_DROPOUT_PROBABILITY 0.0

#define ARRAY_X_DEF Real* gX

#define ARRAY_X gX

#define ARRAY_WEIGHTS_DEF Real* gWKQV, Real* gBKQV, Real* gWO, Real* gBO, Real* gS1, Real* gB1, Real* gLINB1, Real* gLINW1, Real* gS2, Real* gB2, Real* gLINB2, Real* gLINW2
#define ARRAY_WEIGHTS gWKQV, gBKQV, gWO, gBO, gS1, gB1, gLINB1, gLINW1, gS2, gB2, gLINB2, gLINW2

#define ARRAY_FWD_INTERM_DEF Real* gKKQQVV, Real* gWKKWQQWVV, Real* gBETA, Real* gALPHA, Real* gATTN_DROP_MASK, Real* gATTN_DROP, Real* gGAMMA, Real* gATT, Real* gDROP1MASK, Real* gSB1, Real* gSB1_LINW1, Real* gDROP2, Real* gLIN1, Real* gDROP2_LINW2, Real* gLIN2, Real* gLN2, Real* gLN2STD, Real* gLN2DIFF, Real* gDROP2MASK, Real* gDROP3MASK, Real* gLN1, Real* gLN1STD, Real* gLN1DIFF
#define ARRAY_FWD_INTERM gKKQQVV, gWKKWQQWVV, gBETA, gALPHA, gATTN_DROP_MASK, gATTN_DROP, gGAMMA, gATT, gDROP1MASK, gSB1, gSB1_LINW1, gDROP2, gLIN1, gDROP2_LINW2, gLIN2, gLN2, gLN2STD, gLN2DIFF, gDROP2MASK, gDROP3MASK, gLN1, gLN1STD, gLN1DIFF

#define ARRAY_Y_DEF Real* gSB2
#define ARRAY_Y gSB2

#define ARRAY_D_Y_DEF Real* gDSB2
#define ARRAY_D_Y gDSB2

#define ARRAY_D_WEIGHTS_DEF Real* gDWKQV, Real* gDBKQV, Real* gDWO, Real* gDBO, Real* gDS1, Real* gDB1, Real* gDLINB1, Real* gDLINW1, Real* gDS2, Real* gDB2, Real* gDLINB2, Real* gDLINW2
#define ARRAY_D_WEIGHTS gDWKQV, gDBKQV, gDWO, gDBO, gDS1, gDB1, gDLINB1, gDLINW1, gDS2, gDB2, gDLINB2, gDLINW2

#define ARRAY_BWD_INTERM_DEF Real* gDLN2, Real* gDRESID2, Real* gDLIN2, Real* gDDROP2, Real* gDLIN1, Real* gDLIN1_LINW1, Real* gDLN1, Real* gDRESID1, Real* gDATT, Real* gDXATT, Real* gDGAMMA, Real* gDATTN_DROP, Real* gDBETA, Real* gDKKQQVV
#define ARRAY_BWD_INTERM gDLN2, gDRESID2, gDLIN2, gDDROP2, gDLIN1, gDLIN1_LINW1, gDLN1, gDRESID1, gDATT, gDXATT, gDGAMMA, gDATTN_DROP, gDBETA, gDKKQQVV

#define ARRAY_D_X_DEF Real* gDX
#define ARRAY_D_X gDX


#define ENCODER_FORWARD_DEF                     ARRAY_X_DEF,                     ARRAY_WEIGHTS_DEF,                     ARRAY_FWD_INTERM_DEF,                     ARRAY_Y_DEF

#define ENCODER_BACKWARD_DEF                     ARRAY_D_Y_DEF,                     ARRAY_D_WEIGHTS_DEF,                     ARRAY_BWD_INTERM_DEF,                     ARRAY_WEIGHTS_DEF,                     ARRAY_FWD_INTERM_DEF,                     ARRAY_X_DEF,                     ARRAY_D_X_DEF

#define ENCODER_FORWARD                     ARRAY_X,                     ARRAY_WEIGHTS,                     ARRAY_FWD_INTERM,                     ARRAY_Y

#define ENCODER_BACKWARD                     ARRAY_D_Y,                     ARRAY_D_WEIGHTS,                     ARRAY_BWD_INTERM,                     ARRAY_WEIGHTS,                     ARRAY_FWD_INTERM,                     ARRAY_X,                     ARRAY_D_X

struct dimB { enum { value = sizeB }; };
struct dimK { enum { value = sizeS }; };
struct dimJ { enum { value = sizeS }; };
struct dimH { enum { value = sizeH }; };
struct dimP { enum { value = sizeP }; };
struct dimI { enum { value = sizeI }; };
struct dimU { enum { value = sizeU }; };
struct dimQ { enum { value = 3 }; };

using lKKQQVV = metal::list<dimQ, dimH, dimB, dimP, dimJ>;
using lWKKWQQWVV = metal::list<dimQ, dimP, dimH, dimB, dimJ>;
using lBETA = metal::list<dimH, dimB, dimJ, dimK>;
using lALPHA = metal::list<dimH, dimB, dimJ, dimK>;
using lATTN_DROP_MASK = metal::list<dimH, dimB, dimJ, dimK>;
using lATTN_DROP = metal::list<dimH, dimB, dimJ, dimK>;
using lGAMMA = metal::list<dimP, dimH, dimB, dimJ>;
using lATT = metal::list<dimB, dimJ, dimI>;
using lDROP1MASK = metal::list<dimB, dimJ, dimI>;
using lSB1 = metal::list<dimB, dimJ, dimI>;
using lSB1_LINW1 = metal::list<dimB, dimJ, dimU>;
using lDROP2 = metal::list<dimB, dimJ, dimU>;
using lLIN1 = metal::list<dimB, dimJ, dimU>;
using lDROP2_LINW2 = metal::list<dimB, dimJ, dimI>;
using lLIN2 = metal::list<dimJ, dimB, dimI>;
using lLN2 = metal::list<dimJ, dimB, dimI>;
using lLN2STD = metal::list<dimB, dimJ>;
using lLN2DIFF = metal::list<dimB, dimJ, dimI>;
using lDROP2MASK = metal::list<dimB, dimJ, dimU>;
using lDROP3MASK = metal::list<dimB, dimJ, dimI>;
using lLN1 = metal::list<dimB, dimJ, dimI>;
using lLN1STD = metal::list<dimB, dimJ>;
using lLN1DIFF = metal::list<dimB, dimJ, dimI>;
using lDLN2 = metal::list<dimJ, dimB, dimI>;
using lDRESID2 = metal::list<dimI, dimB, dimJ>;
using lDLIN2 = metal::list<dimB, dimJ, dimI>;
using lDDROP2 = metal::list<dimU, dimB, dimJ>;
using lDLIN1 = metal::list<dimU, dimB, dimJ>;
using lDLIN1_LINW1 = metal::list<dimI, dimB, dimJ>;
using lDLN1 = metal::list<dimI, dimB, dimJ>;
using lDRESID1 = metal::list<dimI, dimB, dimJ>;
using lDATT = metal::list<dimI, dimB, dimJ>;
using lDXATT = metal::list<dimB, dimJ, dimI>;
using lDGAMMA = metal::list<dimP, dimH, dimB, dimJ>;
using lDATTN_DROP = metal::list<dimH, dimB, dimJ, dimK>;
using lDBETA = metal::list<dimH, dimB, dimJ, dimK>;
using lDKKQQVV = metal::list<dimQ, dimP, dimH, dimB, dimJ>;
using lWKQV = metal::list<dimQ, dimP, dimH, dimI>;
using lBKQV = metal::list<dimQ, dimP, dimH>;
using lWO = metal::list<dimI, dimP, dimH>;
using lBO = metal::list<dimI>;
using lS1 = metal::list<dimI>;
using lB1 = metal::list<dimI>;
using lLINB1 = metal::list<dimU>;
using lLINW1 = metal::list<dimI, dimU>;
using lS2 = metal::list<dimI>;
using lB2 = metal::list<dimI>;
using lLINB2 = metal::list<dimI>;
using lLINW2 = metal::list<dimI, dimU>;
using lX = metal::list<dimB, dimJ, dimI>;
using lSB2 = metal::list<dimB, dimJ, dimI>;
using lDWKQV = metal::list<dimQ, dimP, dimH, dimI>;
using lDBKQV = metal::list<dimQ, dimP, dimH>;
using lDWO = metal::list<dimI, dimP, dimH>;
using lDBO = metal::list<dimI>;
using lDS1 = metal::list<dimI>;
using lDB1 = metal::list<dimI>;
using lDLINB1 = metal::list<dimU>;
using lDLINW1 = metal::list<dimI, dimU>;
using lDS2 = metal::list<dimI>;
using lDB2 = metal::list<dimI>;
using lDLINB2 = metal::list<dimI>;
using lDLINW2 = metal::list<dimI, dimU>;
using lDX = metal::list<dimB, dimJ, dimI>;
using lDSB2 = metal::list<dimB, dimJ, dimI>;
using lWKKself = metal::list<dimP, dimH, dimB, dimJ>;
using lWQQself = metal::list<dimP, dimH, dimB, dimJ>;
using lWVVself = metal::list<dimP, dimH, dimB, dimJ>;
using lBK = metal::list<dimP, dimH>;
using lBQ = metal::list<dimP, dimH>;
using lBV = metal::list<dimP, dimH>;
using lDBK = metal::list<dimP, dimH>;
using lDBQ = metal::list<dimP, dimH>;
using lDBV = metal::list<dimP, dimH>;
using lQ = metal::list<dimB, dimJ, dimI>;
using lK = metal::list<dimB, dimK, dimI>;
using lV = metal::list<dimB, dimK, dimI>;
using lKK = metal::list<dimH, dimB, dimP, dimK>;
using lQQ = metal::list<dimH, dimB, dimP, dimJ>;
using lVV = metal::list<dimH, dimB, dimP, dimK>;
using lDKK = metal::list<dimP, dimH, dimB, dimK>;
using lDQQ = metal::list<dimP, dimH, dimB, dimJ>;
using lDVV = metal::list<dimP, dimH, dimB, dimK>;
using lKKself = metal::list<dimH, dimB, dimP, dimJ>;
using lQQself = metal::list<dimH, dimB, dimP, dimJ>;
using lVVself = metal::list<dimH, dimB, dimP, dimJ>;
using lDKKself = metal::list<dimP, dimH, dimB, dimJ>;
using lDQQself = metal::list<dimP, dimH, dimB, dimJ>;
using lDVVself = metal::list<dimP, dimH, dimB, dimJ>;
using lWKK = metal::list<dimP, dimH, dimB, dimJ>;
using lWQQ = metal::list<dimP, dimH, dimB, dimJ>;
using lWVV = metal::list<dimP, dimH, dimB, dimJ>;
using lWK = metal::list<dimP, dimH, dimI>;
using lWQ = metal::list<dimP, dimH, dimI>;
using lWV = metal::list<dimP, dimH, dimI>;
using lDWK = metal::list<dimP, dimH, dimI>;
using lDWQ = metal::list<dimP, dimH, dimI>;
using lDWV = metal::list<dimP, dimH, dimI>;


#define algoWKKWQQWVV CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define algoBETA CUBLAS_GEMM_ALGO1_TENSOR_OP
#define algoGAMMA CUBLAS_GEMM_ALGO2_TENSOR_OP
#define algoATT CUBLAS_GEMM_ALGO3_TENSOR_OP
#define algoSB1_LINW1 CUBLAS_GEMM_ALGO3_TENSOR_OP
#define algoDROP2_LINW2 CUBLAS_GEMM_ALGO6_TENSOR_OP
#define algoDDROP2 CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define algoDLINW2 CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define algoDLIN1_LINW1 CUBLAS_GEMM_ALGO6_TENSOR_OP
#define algoDLINW1 CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define algoDGAMMA CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define algoDWO CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define algoDATTN_DROP CUBLAS_GEMM_ALGO9_TENSOR_OP
#define algoDVV CUBLAS_GEMM_ALGO10_TENSOR_OP
#define algoDQQ CUBLAS_GEMM_ALGO10_TENSOR_OP
#define algoDKK CUBLAS_GEMM_ALGO2_TENSOR_OP
#define algoDXATT CUBLAS_GEMM_ALGO6_TENSOR_OP
#define algoDWKQV CUBLAS_GEMM_DEFAULT_TENSOR_OP


#define sdAIB_DV dimB
#define sdAIB_DT dimJ
#define sdSM_DV dimK
#define sdBDRLN1_DV dimI
#define sdBAD_DV dimU
#define sdBAD_DT dimU
#define sdBDRLN2_DV dimI
#define sdBSB_DV dimI
#define sdBSB_DW dimB
#define sdBLNRD1_DV dimJ
#define sdBDRLB_DV dimU
#define sdBDRLB_DW dimJ
#define sdEBSB_DV dimJ
#define sdEBSB_DW dimJ
#define sdBLNRD2_DV dimJ
#define sdBAOB_DV dimJ
#define sdBAOB_DW dimB
#define sdBS_DV dimK
#define sdBAIB_DV dimJ
#define sdBAIB_DW dimB
#define sdBEI_DT dimI
#define sdBEI_DV dimJ


